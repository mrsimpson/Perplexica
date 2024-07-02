import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { PromptTemplate } from '@langchain/core/prompts';
import formatChatHistoryAsString from '../utils/formatHistory';
import { BaseMessage } from '@langchain/core/messages';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { searchSearxng } from '../lib/searxng';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';

const imageSearchChainPrompt = `
Du bekommst unten ein Gespräch und eine Folgefrage. Du musst die Folgefrage so umformulieren, dass sie eine eigenständige Frage ist, die vom LLM verwendet werden kann, um im Web nach Bildern zu suchen.
Du musst sicherstellen, dass die umformulierte Frage mit dem Gespräch übereinstimmt und relevant für das Gespräch ist.

---
Beispiele:
1. Folgefrage: Was ist eine Katze?
Umformuliert: Eine Katze

2. Folgefrage: Was ist ein Auto? Wie funktioniert es?
Umformuliert: Automechanik

3. Folgefrage: Wie funktioniert eine Klimaanlage?
Umformuliert: Klimaanlage Funktion

Gespräch:
{chat_history}

Folgefrage: {query}
Umformulierte Frage:
---

Antworte bitte auf Deutsch.
`;

type ImageSearchChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const strParser = new StringOutputParser();

const createImageSearchChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    RunnableMap.from({
      chat_history: (input: ImageSearchChainInput) => {
        return formatChatHistoryAsString(input.chat_history);
      },
      query: (input: ImageSearchChainInput) => {
        return input.query;
      },
    }),
    PromptTemplate.fromTemplate(imageSearchChainPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      const res = await searchSearxng(input, {
        engines: ['bing images', 'google images'],
      });

      const images = [];

      res.results.forEach((result) => {
        if (result.img_src && result.url && result.title) {
          images.push({
            img_src: result.img_src,
            url: result.url,
            title: result.title,
          });
        }
      });

      return images.slice(0, 10);
    }),
  ]);
};

const handleImageSearch = (
    input: ImageSearchChainInput,
    llm: BaseChatModel,
) => {
  const imageSearchChain = createImageSearchChain(llm);
  return imageSearchChain.invoke(input);
};

export default handleImageSearch;
