import Anthropic from '@anthropic-ai/sdk'
import type { Suggestions, Summary } from '@/lib/types'

if (!process.env.ANTHROPIC_API_KEY) {
  throw new Error('ANTHROPIC_API_KEY is required but not set in environment variables')
}

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

const LIVE_SYSTEM_PROMPT = `Tu es un extracteur temps réel d'informations de réunion.
Rôle: transformer des extraits de transcription en JSON exploitable pour l'UI.
Exigences:
- Ne répondre QU'EN JSON conforme au schéma LIVE.
- Zéro texte hors JSON, zéro explication.
- Si l'extrait est bruité ou peu informatif, renvoyer des tableaux vides.
- Ne pas halluciner d'assignations ou de dates si non mentionnées.
- Détecter et ignorer les apartés sans rapport (blagues, bruits).`

const FINAL_SYSTEM_PROMPT = `Tu es un synthétiseur expert de réunions. Tu produis des rapports détaillés, approfondis et actionnables.
Exigences:
- Sortie stricte en JSON conforme au schéma FINAL.
- Zéro texte hors JSON.
- Pas d'hallucinations: s'abstenir quand l'info n'existe pas.
- Écrire en français s'il y a majoritairement du FR, sinon en anglais.
- Style: clair, factuel, professionnel, avec beaucoup de détails et de contexte.
- Pour la vue synthétique: des résumés courts et clairs.
- Pour la vue détaillée: des analyses approfondies avec beaucoup de contexte, des explications détaillées et des nuances.
- IMPORTANT: Même si la conversation est courte ou informelle, analyse le contenu et produis un résumé basé sur ce qui a été dit. Ne pas dire "Aucun contenu" si du texte est fourni.`

export async function analyzeLiveTranscript(
  transcript: string
): Promise<Suggestions> {
  try {
    const userPrompt = `TRANSCRIPT_RAW:
${transcript}

Contrainte:
- Si un même item est répété, reviens avec la version la plus courte et la plus claire.
- Utilise "confidence" ∈ [0.0,1.0] pour "decisions" et "actions".

Réponds EXCLUSIVEMENT avec un JSON valide selon ce schéma:
{
  "topics": ["string"],
  "decisions": [{"text":"string","confidence":0.0}],
  "actions": [{"text":"string","assignee":"string|null","due_date":"YYYY-MM-DD|null","confidence":0.0}]
}`

    const message = await anthropic.messages.create({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 1024,
      temperature: 0.2,
      system: LIVE_SYSTEM_PROMPT,
      messages: [
        {
          role: 'user',
          content: userPrompt,
        },
      ],
    })

    const content = message.content[0]
    if (content.type !== 'text') {
      throw new Error('Unexpected response type')
    }

    console.log('DEBUG - Claude raw response (full):', JSON.stringify(content.text))

    // Parse JSON response
    const jsonMatch = content.text.match(/\{[\s\S]*\}/)
    if (!jsonMatch) {
      console.log('DEBUG - No JSON found in response')
      return { topics: [], decisions: [], actions: [] }
    }

    const parsed = JSON.parse(jsonMatch[0])
    console.log('DEBUG - Parsed suggestions (full):', JSON.stringify(parsed))

    return {
      topics: parsed.topics || [],
      decisions: parsed.decisions || [],
      actions: parsed.actions || [],
    }
  } catch (error) {
    console.error('Error analyzing live transcript:', error)
    return { topics: [], decisions: [], actions: [] }
  }
}

export async function generateFinalSummary(
  transcript: string[],
  validatedSuggestions?: Suggestions,
  userNotes?: string,
  templateStructure?: any
): Promise<Summary> {
  try {
    const fullTranscript = transcript.join('\n')

    let userPrompt = `TRANSCRIPT_COMPLET:
${fullTranscript}`

    if (validatedSuggestions) {
      userPrompt += `

LISTES_VALIDÉES_PAR_UTILISATEUR:
${JSON.stringify(validatedSuggestions, null, 2)}`
    }

    if (userNotes && userNotes.trim().length > 0) {
      // Strip HTML tags for better analysis
      const notesText = userNotes.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()
      userPrompt += `

NOTES_PRISES_PAR_L_UTILISATEUR:
${notesText}

IMPORTANT: L'utilisateur a pris ces notes pendant la réunion. Tu dois:
1. Utiliser ces notes pour enrichir ton analyse et ton résumé
2. Intégrer les points clés des notes dans le résumé final
3. Utiliser ces notes comme base pour "enhancedNotes" en y ajoutant le contexte de la transcription`
    }

    // If a template is provided, use its custom prompt and structure
    if (templateStructure?.prompt) {
      userPrompt += `

${templateStructure.prompt}`
    } else {
      // Default prompt
      userPrompt += `

Contrainte:
- Prioriser les éléments VALIDÉS si fournis.
- Pour la VUE SYNTHÉTIQUE (summary, actions, decisions, open_questions, topics):
  * "summary": 2-3 phrases claires résumant l'essence de la réunion
  * "topics": identifier les grands sujets avec une synthèse courte (1-2 lignes) pour chacun
  * "actions": liste claire et concise des actions à suivre
  * "decisions": liste claire des décisions prises
  * "open_questions": liste des questions ouvertes

- Pour la VUE DÉTAILLÉE (detailed):
  * "summary_detailed": un résumé de 2-3 paragraphes décrivant le contexte, les enjeux et conclusions
  * "actions_detailed": un paragraphe (5-8 lignes) expliquant le contexte des actions et priorités
  * "decisions_detailed": un paragraphe (5-8 lignes) expliquant le contexte des décisions et implications
  * "open_questions_detailed": un paragraphe (5-8 lignes) expliquant les questions ouvertes et leur importance
  * "topics_detailed": pour chaque sujet, un paragraphe (8-10 lignes) couvrant le contexte, discussions principales et conclusions

- Pour les NOTES ENRICHIES (enhancedNotes):
  * TOUJOURS générer des notes enrichies, même sans notes utilisateur
  * Si notes utilisateur: les enrichir avec le contexte de la transcription
  * Si pas de notes utilisateur: créer des notes structurées basées sur la transcription
  * Format HTML avec structure claire:
    - <h2> pour les grands thèmes
    - <h3> pour les sous-sections
    - <ul> et <li> pour les listes à puces
    - <strong> pour les points importants
    - <em> pour les nuances ou détails
    - <p> pour les paragraphes explicatifs
  * Contenu: résumé narratif et chronologique de la réunion avec tous les détails importants, citations clés, contexte et décisions
  * Style: notes de réunion professionnelles, détaillées et bien structurées

Réponds EXCLUSIVEMENT avec un JSON valide selon ce schéma:
{
  "summary": "string (synthèse courte en 2-3 phrases)",
  "actions": [{"text":"string","assignee":"string|null","due_date":"YYYY-MM-DD|null","priority":"low|medium|high|null"}],
  "decisions": [{"text":"string"}],
  "open_questions": ["string (questions ouvertes et follow-ups)"],
  "topics": [{"title":"string","summary":"string (synthèse courte de 1-2 lignes)"}],
  "detailed": {
    "summary_detailed": "string (résumé de 2-3 paragraphes)",
    "actions_detailed": "string (paragraphe de 5-8 lignes)",
    "decisions_detailed": "string (paragraphe de 5-8 lignes)",
    "open_questions_detailed": "string (paragraphe de 5-8 lignes)",
    "topics_detailed": [{"title":"string","detailed_summary":"string (paragraphe de 8-10 lignes)"}]
  },
  "enhancedNotes": "string (notes enrichies structurées en HTML avec h2/h3/ul/li/strong/em/p - TOUJOURS générer)"
}`
    }

    const isLongTranscript = fullTranscript.length > 15000 // ~15k characters
    const model = isLongTranscript ? 'claude-sonnet-4-6' : 'claude-haiku-4-5-20251001'
    const maxTokens = isLongTranscript ? 16000 : 4096

    if (isLongTranscript) {
      console.log(`⚠️ Long transcript detected (${fullTranscript.length} chars), using Sonnet 4.6 with extended output`)
    }

    const message = await anthropic.messages.create({
      model,
      max_tokens: maxTokens,
      temperature: 0.3,
      system: FINAL_SYSTEM_PROMPT,
      messages: [
        {
          role: 'user',
          content: userPrompt,
        },
      ],
    })

    const content = message.content[0]
    if (content.type !== 'text') {
      throw new Error('Unexpected response type')
    }

    // Use robust JSON parsing with sanitization
    const { robustJSONParse } = await import('@/lib/utils/sanitize')
    const parsed = robustJSONParse(content.text)
    return {
      summary: parsed.summary || '',
      actions: parsed.actions || [],
      decisions: parsed.decisions || [],
      open_questions: parsed.open_questions || [],
      topics: parsed.topics || [],
      detailed: parsed.detailed || undefined,
      rawNotes: userNotes || undefined,
      enhancedNotes: parsed.enhancedNotes || undefined,
      // Keep deprecated fields for backward compatibility
      highlights: parsed.highlights || [],
      risks: parsed.risks || [],
      next_steps: parsed.next_steps || [],
    }
  } catch (error) {
    console.error('Error generating final summary:', error)
    throw error
  }
}
