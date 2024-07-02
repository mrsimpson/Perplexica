import { RunnableSequence, RunnableMap } from '@langchain/core/runnables';
import ListLineOutputParser from '../lib/outputParsers/listLineOutputParser';
import { PromptTemplate } from '@langchain/core/prompts';
import formatChatHistoryAsString from '../utils/formatHistory';
import { BaseMessage } from '@langchain/core/messages';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { ChatOpenAI } from '@langchain/openai';

const suggestionGeneratorPrompt = `
Du bist ein KI-Vorschlagsgenerator für eine KI-gestützte Suchmaschine. Dir wird unten ein Gespräch gegeben. Du musst basierend auf dem Gespräch 4-5 Vorschläge generieren. Die Vorschläge sollten relevant für das Gespräch sein und vom Benutzer verwendet werden können, um das Chat-Modell nach weiteren Informationen zu fragen.
Stelle sicher, dass die Vorschläge relevant für das Gespräch und für den Benutzer hilfreich sind. Beachte, dass der Benutzer diese Vorschläge verwenden könnte, um das Chat-Modell nach weiteren Informationen zu fragen.
Stelle sicher, dass die Vorschläge mittel in der Länge, informativ und relevant für das Gespräch sind.

Antworte bitte auf Deutsch.

Gib diese Vorschläge getrennt durch Zeilenumbrüche zwischen den XML-Tags <suggestions> und </suggestions> an. Zum Beispiel:

<suggestions>
Erzähl mir mehr über SpaceX und ihre aktuellen Projekte
Was sind die neuesten Nachrichten über SpaceX?
Wer ist der CEO von SpaceX?
</suggestions>

Conversation:
{chat_history}
`;

type SuggestionGeneratorInput = {
  chat_history: BaseMessage[];
};

const outputParser = new ListLineOutputParser({
  key: 'suggestions',
});

const createSuggestionGeneratorChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    RunnableMap.from({
      chat_history: (input: SuggestionGeneratorInput) =>
          formatChatHistoryAsString(input.chat_history),
    }),
    PromptTemplate.fromTemplate(suggestionGeneratorPrompt),
    llm,
    outputParser,
  ]);
};

const generateSuggestions = (
    input: SuggestionGeneratorInput,
    llm: BaseChatModel,
) => {
  (llm as ChatOpenAI).temperature = 0;
  const suggestionGeneratorChain = createSuggestionGeneratorChain(llm);
  return suggestionGeneratorChain.invoke(input);
};

export default generateSuggestions;
