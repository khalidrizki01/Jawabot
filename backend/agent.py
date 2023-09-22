from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.tools.google_serper.tool import GoogleSerperRun
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import MessagesPlaceholder
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)

PREFIX = ("Jawabot is a specialized chatbot designed to facilitate Javanese language learning for non-Javanese speakers. "
"It engages in conversations and provides responses in the Javanese language  to help people grow accustomed to speaking Javanese. " 
"Even if the user uses Bahasa Indonesia or English, Jawabot replies exclusively in Javanese")

def split_texts(text_name):
  loader = TextLoader(text_name, encoding="utf-8")
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  texts = text_splitter.split_documents(documents)
  return texts

def get_or_create_vectorstore(id, data_src, embeddings):
    persist_directory = f"db_{id}"
    if not os.path.exists(persist_directory):
        return Chroma.from_documents(split_texts(data_src), embeddings, persist_directory=persist_directory,collection_name=f"{id}-col")
    else: 
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=f"{id}-col")


llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.0)
embeddings = OpenAIEmbeddings()

swara_sigeg_docsearch = get_or_create_vectorstore("swara_sigeg", "data/widya-swara-lan-sigeg.txt", embeddings)  #Chroma.from_documents(split_texts("data/widya-swara-lan-sigeg.txt"), embeddings, persist_directory='db_swara',collection_name="swara-sigeg-col")
unggah_docsearch = get_or_create_vectorstore("unggah", "data/C-unggah-ungguh-basa.txt", embeddings)  #Chroma.from_documents(split_texts("data/C-unggah-ungguh-basa.txt"), embeddings,persist_directory='db_unggah',collection_name="unggah-ungguh-col")
geguritan_docsearch = get_or_create_vectorstore("geguritan", "data/E-geguritan.txt", embeddings)  #Chroma.from_documents(split_texts("data/E-geguritan.txt"), embeddings,persist_directory='db_geguritan', collection_name="geguritan-col")
widyatembung_docsearch = get_or_create_vectorstore("tembung", "data/F-widya-tembung.txt", embeddings)  #Chroma.from_documents(split_texts("data/F-widya-tembung.txt"), embeddings,persist_directory='db_tembung', collection_name="widya-tembung-col")
widyaukara_docsearch = get_or_create_vectorstore("ukara", "data/G-widya-ukara.txt", embeddings)  #Chroma.from_documents(split_texts("data/G-widya-ukara.txt"), embeddings,persist_directory='db_ukara', collection_name="widya-ukara-col")
lagon_docsearch = get_or_create_vectorstore("lagon", "data/H-lagon-dolanan.txt", embeddings)  #Chroma.from_documents(split_texts("data/H-lagon-dolanan.txt"), embeddings,persist_directory='db_lagon', collection_name="lagon-dolanan-col")
macapat_docsearch = get_or_create_vectorstore("macapat", "data/I-tembang-macapat.txt", embeddings)  #Chroma.from_documents(split_texts("data/I-tembang-macapat.txt"), embeddings,persist_directory='db_macapat', collection_name="tembang-macapat-col")

contoh = [
    {
        "question" : "Jelaske apa iku widya swara kanthi ringkes",
        "answer" : "Widya swara yaiku ilmu sing ngebahas lan ngrasakake swara. Swara sing dibahas yaiku swara vokal basa Jawa."
    },
    {
        "question": "Apa wae jinis vokal ing basa jawa?",
        "answer" : "Jinis vokal ing basa Jawa ana sepuluh, yaiku a jejeg, a miring, i jejeg, i miring, u jejeg, u miring, ê, é, o jejeg, lan o miring."
    },
    {
        "question": "Kepiye cara ngucapake saben vokal basa Jawa?",
        "answer": 
        """1. A jejeg dimaca /ͻ/, contone apa [ͻpͻ] lan segara [sӘgͻnͻ]. 
        2. A miring dimaca /a/, contone aku [aku] lan alas [alas]. 
        3. I jejeg dimaca /i/, contone isin [isin]. 
        4. I miring dibaca /é/, contone pitik [pitIk], lan cuwil [cuwIl]. 
        5. U jejeg dimaca /u/, contone urip [urIp]. 
        6. U miring dimaca /o/, ana ing tengahing tembung. Contone gunung [gunUɳ] lan kasur [kasUr]. 
        7. Conto vokal /ê/ yaiku emoh [Әmͻh]. 
        8. Conto vokal /é/ yaiku eman [eman]. 
        9. Vokal /o/ jejeg dimaca /o/, contone omah [omah]. 
        10. Vokal o miring dimaca /ɔ/, ana ing ngarep lan tengahing tembung. Contone vokal o miring yaiku orong-orong [ɔrɔŋ- ɔrɔŋ] lan obor [ɔbɔr]"""
    },
    {
        "question" : "Sebutna conto tembung sing nduwe pocapan beda antara dialek Yogya karo Surabayan",
        "answer": "yen tembung disusun karo loro wanda lan saben wanda ngandhut vokal /i/, vokal /i/ wanda sing ngarep dimaca jejeg ing dialek Jogja/Sala. Yen ing dialek Surabayan, wanda sing ngarep dimaca miring. Hal sing mirip uga kejadian yen ana tembung sing disusun karo loro wanda lan saben wanda ngandhut vokal /u/. Vokal u wanda sing ngarap dimaca jejeg ing dialek Jogja/Sala. Ing dialek Surabayan, vokal /i/ kuwi dimaca miring"
    }
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
        ("ai", "{answer}")
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=contoh
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Kowe agen pengambil dokumen babagan widya swara lan widya sigeg (fonetik Jawa). "
                     "Gunakake potongan konteks ing ngisor iki kanggo mangsuli pitakon ing pungkasan. "
                     "Yen sampeyan ora ngerti jawaban, mung ngomong yen sampeyan ora ngerti, aja nyoba nggawe jawaban. "
                     "{context}"),
        few_shot_prompt,
        ("human", "{question}"),
    ]
)

swara_sigeg = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.0), 
    chain_type="stuff", 
    retriever=swara_sigeg_docsearch.as_retriever(), 
    chain_type_kwargs={
        "prompt":final_prompt
    }
)

# swara_sigeg = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.0), chain_type="stuff", retriever=swara_sigeg_docsearch.as_retriever())
unggah = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.0), chain_type="stuff", retriever=unggah_docsearch.as_retriever())
geguritan = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.0), chain_type="stuff", retriever=geguritan_docsearch.as_retriever())
widyatembung = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.0), chain_type="stuff", retriever=widyatembung_docsearch.as_retriever())
widyaukara = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.0), chain_type="stuff", retriever=widyaukara_docsearch.as_retriever())
lagon = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.0), chain_type="stuff", retriever=lagon_docsearch.as_retriever())
macapat = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.0), chain_type="stuff", retriever=macapat_docsearch.as_retriever())

tools = [
    Tool(
        name='Widya-Swara',
        func=swara_sigeg.run,
        description=("A document list concerning 'Widya Swara' and 'Widya Sigeg', which is Javanese phonetics. "
                    #  "It also contains other subjects related to it such as "
                    # "dialects and differences in meaning between words that uses similar albeit different vocal letters. "
                    # "If you don't know the answer, you may use other tools to find the correct answer. "
                    # "Input should be as clear as possible with detailed context. "
                    "Example input 'Sebutna apa wae jinis vokal ing basa jawa karo contone!', "
                    # "'Terangna carane ngucapake saben jinis vokal ing basa jawa', "
                    "'Terangna bedane dialek Yogya/Sala karo dialek Surabayan', "
                    "'Terangna bedane sêsêk karo sésék', and 'Tulisna cara nulis beduk sing bener ing basa jawa'.")
    ),
    Tool(
        name='Unggah-Ungguh',
        func=unggah.run,
        description=("A document list concerning 'unggah-ungguh' which is Javanese way of speaking "
                    "which includes social norms, speaking ethics, and ways of speaking according to the situation and relationship between speakers. "
                    # "If you don't know the answer, you may use other tools to find the correct answer. "
                    # "Input should be as clear as possible with detailed context. "
                    "Example input 'Sebutna jumlah unggah-ungguh miturut kosa kata', "
                    "'Sebutna kapan wae kudu nganggo basa ngoko alus', and "
                    "'Sebutna apa wae ater-ater lan panambang ing basa krama.' "
                )
    ),
    Tool(
        name='Geguritan',
        func=geguritan.run,
        description=("A document list concerning Javanese modern poetry (geguritan). "
                    # "If you don't know the answer, you may use other tools to find the correct answer. "
                    # "Input should be as clear as possible with detailed context. "
                    "Example input 'Sebutna unsur-unsur geguritan', "
                    "'Sebutna apa wae jinis purwakanthi', and "
                    "'Gawena geguritan babagan ibu' "
                )
    ),
    Tool(
        name='Widya-Tembung',
        func=widyatembung.run,
        description=("A document list concerning 'Widya Tembung' which is Javanese morphology. "
                    # "If you don't know the answer, you may use other tools to find the correct answer. "
                    # "Input should be as clear as possible with detailed context. "
                    "Example input 'Terangna apa iku ater-ater, seselan, panambang, lan bebarengan?', "
                    "'Sebutna jinis tembung ing basa jawa', and "
                    "'Miturut wujude, tembung kaperang dadi apa wae?' "
                )
    ),
    Tool(
        name='Widya-Ukara',
        func=widyaukara.run,
        description=("A document list concerning 'Widya Ukara' which is Javanese syntax. "
                    # "If you don't know the answer, you may use other tools to find the correct answer. "
                    # "Input should be as clear as possible with detailed context. "
                    "Example input 'Sebutna jinis-jinis struktur ukara ing basa jawa', "
                    "'Tunjukna pola ukarane ukara ing ngisor iki', and "
                    "'Terangna bedane geganep karo lesan' "
                )
    ),
    Tool(
        name='Lagon-Dolanan',
        func=lagon.run,
        description=("A document list concerning 'Lagon Dolanan' which is Javanese children's play song. "
                    # "If you don't know the answer, you may use other tools to find the correct answer. "
                    # "Input should be as clear as possible with detailed context. "
                    "Example input 'Sebutna aturan lagon dolanan', and "
                    "'Gawena lagon dolanan babagan gajah' "
                )
    ),
    Tool(
        name='Tembang-Macapat',
        func=macapat.run,
        description=("A document list concerning 'Tembang Macapat', a form of traditional Javanese poetry that follows specific metrical and melodic patterns. "
                    # "If you don't know the answer, you may use other tools to find the correct answer. "
                    # "Input should be as clear as possible with detailed context. "
                    "Example input 'Sebutna jinis-jinis tembang macapat', "
                    "'Sebutna jumlah guru wilangan lan guru lagu tembang maskumambang', and"
                    "'Kepiye watake tembang pocung'"
                )
    ),
    GoogleSerperRun(
        api_wrapper=GoogleSerperAPIWrapper(),
        description= (
            "A low-cost Google Search API. Useful when you need to get accurate information to answer a question. "
            "When the subject is about Javanese linguistics, phonetics, literature, language, societal values, and cultures, "
            "you may only use this tool if the other tools doesn't have the answer. "
            # "You must always use this when you are unsure of your answer or when you need a search for information. "
            "Input should be as concise and as clearly as possible and formatted as a single search term."
        )
    )
]

class JawabotAgent(object):
    def __init__(self):
        self.memory = None
        self.agent_chain = None
#         self.id = str(id)

        self.initialize_agent()

    def initialize_agent(self):

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        agent_kwargs = {
            "system_message" : SystemMessage(content=PREFIX),
            "extra_prompt_messages": [
                MessagesPlaceholder(variable_name="chat_history"),
                SystemMessage(content="Remember, Jawabot replies exclusively in Javanese "
                    "even if the user is speaking in other languages. "
                    "Jawabot may speak in Bahasa Indonesia when and only when the user ask for translation for certain words. "
 #                   "Jawabot should never reply in full Bahasa Indonesia or English."
                    )
                ],
        }

        self.agent_chain = initialize_agent(
                                tools,
                                llm,
                                agent=AgentType.OPENAI_FUNCTIONS,
                                verbose=False,
                                agent_kwargs=agent_kwargs,
                                memory=self.memory
                            )
    # def run(self, text):
    #     return self.agent_chain.run(input=text)

    def async_generate(self, text):
        resp = self.agent_chain.run(input=text)
        return resp