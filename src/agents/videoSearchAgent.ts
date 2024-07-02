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

const VideoSearchChainPrompt = `
  Du bekommst unten ein Gespräch und eine Folgefrage. Du musst die Folgefrage so umformulieren, dass sie eine eigenständige Frage wird, die vom LLM verwendet werden kann, um auf YouTube nach Videos zu suchen.
  Du musst sicherstellen, dass die umformulierte Frage mit dem Gespräch übereinstimmt und zum Gespräch relevant ist.
  
  Beispiel:
  1. Folgefrage: Wie funktioniert ein Auto?
  Umformulierte Frage: Wie funktioniert ein Auto?
  
  2. Folgefrage: Was ist die Relativitätstheorie?
  Umformulierte Frage: Was ist die Relativitätstheorie
  
  3. Folgefrage: Wie funktioniert eine Klimaanlage?
  Umformulierte Frage: Wie funktioniert eine Klimaanlage
  
  Gespräch:
  {chat_history}
  
  Folgefrage: {query}
  Umformulierte Frage:
  ---
  
  Antworte bitte auf Deutsch.
  `;

type VideoSearchChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const strParser = new StringOutputParser();

const createVideoSearchChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    RunnableMap.from({
      chat_history: (input: VideoSearchChainInput) => {
        return formatChatHistoryAsString(input.chat_history);
      },
      query: (input: VideoSearchChainInput) => {
        return input.query;
      },
    }),
    PromptTemplate.fromTemplate(VideoSearchChainPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      const res = await searchSearxng(input, {
        engines: ['youtube'],
      });

      const videos = [];

      res.results.forEach((result) => {
        if (
            result.thumbnail &&
            result.url &&
            result.title &&
            result.iframe_src
        ) {
          videos.push({
            img_src: result.thumbnail,
            url: result.url,
            title: result.title,
            iframe_src: result.iframe_src,
          });
        }
      });

      return videos.slice(0, 10);
    }),
  ]);
};

const handleVideoSearch = (
    input: VideoSearchChainInput,
    llm: BaseChatModel,
) => {
  const VideoSearchChain = createVideoSearchChain(llm);
  return VideoSearchChain.invoke(input);
};

export default handleVideoSearch;
