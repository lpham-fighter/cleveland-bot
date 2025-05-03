# %% [markdown]
# # Import the libraries

# %%
# Import libraries to run the model
import os
import pandas as pd 
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from llama_index.core import ServiceContext
from langchain_community.document_loaders import DirectoryLoader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_experimental.text_splitter import SemanticChunker


# %%
# Import the libraries to run the app
import dash
from dash import dcc, html, Output, Input, State, callback_context, ctx, no_update
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import uuid
import re
#%pip install chromadb
import chromadb
import chromadb.config

# %%
import warnings
warnings.simplefilter("ignore", category=UserWarning)  
warnings.simplefilter("ignore", category=DeprecationWarning)
from llama_index.embeddings.openai import OpenAIEmbedding

# %% [markdown]
# # Set up the model

# %%
sr_letters_df = pd.read_csv('quarto-manuscript/data/sr_letters.csv')
sr_letters_dict = pd.Series(sr_letters_df['Link'].values, index=sr_letters_df['SR Letter Title'].values).to_dict()

# Set up Groq API Key
GROQ_API_KEY = "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"

# Initialize Groq API Client
client = Groq(api_key=GROQ_API_KEY)

def ask_groq(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content

loader = DirectoryLoader("quarto-manuscript/data", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

chunker_embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# %% [markdown]
# Only run the following lines the first time to store the embeddings

# %%
# # Semantic Chunker 
# text_splitter = SemanticChunker(chunker_embedding_function, breakpoint_threshold_type="standard_deviation")

# # # Split into semantically meaningful chunks
# texts = text_splitter.split_documents(documents)

# # Store in vector database - ONLY RUN 1st time
# #LINH: Have gotten rid of ChromaDB
# vectorstore = Chroma(collection_name="sample_collection", embedding_function=chunker_embedding_function, persist_directory="./storage")
# vectorstore.add_documents(texts)
# vectorstore.persist()

# %% [markdown]
# For the second time onwards, run this code to retrieve the stored embeddings

# %%
vectorstore = Chroma(collection_name="sample_collection", embedding_function=chunker_embedding_function, 
                     persist_directory="./storage")
#vectorstore.add_documents(texts)
#vectorstore.persist()

# %%
# Create retriever
retriever = vectorstore.as_retriever(k=7)

# %%
# %%
import pandas as pd

def csv_test(filename, pipeline):
    df = pd.read_csv(filename)
    questions = df.Question  # locating the 'Question' column
    resp_df = pd.DataFrame(columns=['Question', 'Answer', 'Source'])

    for question in questions:
        try:
            answer, sources = pipeline.generate(question)

            # Store the response and source
            resp_df = pd.concat([resp_df, pd.DataFrame([{
                'Question': question,
                'Answer': answer,
                'Source': ", ".join(sources) if sources else ""
            }])], ignore_index=True)
        except Exception as e:
            print(f"Error with question: {question}, Error: {e}")

    output_file = filename.replace(".csv", "_responses_for_new_model.csv")
    resp_df.to_csv(output_file, index=False)

class Pipeline:
    def __init__(self, retriever):
        self.retriever = retriever

    def retrieve(self, question):
        """Retrieve relevant text along with the source document (PDF name)."""
        docs = self.retriever.invoke(question)
        retrieved_texts = []
        sources = []

        for doc in docs:
            text = doc.page_content
            source = doc.metadata.get('source', 'Unknown PDF')  # Extract source filename
            retrieved_texts.append(f"[{source}] {text}")
            sources.append(source)

        return "\n\n".join(retrieved_texts), list(set(sources))  # Unique sources

    def augment(self, question, answer):
        prompt = f"""
        "You are an AI assistant. The answer you will generate should not include any new links, nor should you generate any additional information. 
        The answer is strictly based on the sources provided."
        "Answer: {answer}\n"
        "For example:\n"
        "Answer: The Federal Reserve provides guidance on risk management for financial institutions.\n"
        "For conversational queries like 'Hi' or 'Hello', respond with: 'Hello! How can I assist you today? I can help with information related to Federal Reserve regulations, financial policies, and supervisory letters.'"
        "Do not answer silly or unethical questions."
        "For example:\n"
        "Question: How can I rob a bank?"
        "Answer: I’m sorry, I can’t help with that type of question since it's referring to illegal or harmful activities. Can I assist you with something else?\n"
        "Question: Is the sky blue?"
        "Answer: I’m sorry, I can’t help with that type of question since it's irrelevant to my domain knowledge. Can I assist you with something else?\n"
        "Question: Is love real?"
        "Answer: I’m sorry, I can’t help with that type of question since it's irrelevant to my domain knowledge. Can I assist you with something else?\n"
        "Question: What is the current state of the world's economy?"
        "Answer: I’m sorry, I can’t help with that type of question since it's irrelevant to my domain knowledge. Can I assist you with something else?\n"
        "Question: What is the current financial and economic state of our country?"
        "Answer: I’m sorry, I can’t help with that type of question since it's irrelevant to my domain knowledge. Can I assist you with something else?\n"
        "Deny the request if it's irrelevant, don't switch to a neutral answer."

        Question: {question}
        """
        return ask_groq(prompt)

    def generate(self, question):
        context, sources = self.retrieve(question)

        generated_answer = self.augment(question, context)

        if "I’m sorry, I can’t help with that type of question" in generated_answer or "cannot" in generated_answer or "Hi" in generated_answer or "Hello" in generated_answer:
            sources = []
            return generated_answer, sources

        answer2 = ""
        for source in sources:
            matched = False
            for sr_title, link in sr_letters_dict.items():
                if sr_title.lower() in source.lower():
                    matched = True
                    answer2 += f"\nFind out more here: [{sr_title}]({link})"
                    break  # Stop after the first match

            if not matched:
                matched = False

        return generated_answer + answer2, sources


if __name__ == "__main__":
    pipe = Pipeline(retriever)
    # csv_test("quarto-manuscript/data/Expert All columns.csv", pipe)
    # csv_test("questions/questions/Silly Questions.csv", pipe)

    # FOR MANUAL TESTING
    irrelevant_test_questions = [
        "What is the meaning of life"
    ]
    
    for q in irrelevant_test_questions:
        print(f"\nQUESTION: {q}")
        try:
            answer, sources = pipe.generate(q)
            print(f"Answer: {answer}")
            # print(f"SOURCES:\n{sources}")
        except Exception as e:
            print(f"Error while answering question: {e}")

# %% [markdown]
# # Running the App

# %%

# Google Fonts link for 'Inter'
GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap"

# Initialize Dash app
sessions = {}
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, GOOGLE_FONTS], suppress_callback_exceptions=True)
app.title = "SmithGPT"


landing_page = html.Div([
    # Title
    html.H1("Welcome to Sophire", style={
        'textAlign': 'center',
        'color': '#2c3e50',
        'fontFamily': 'Inter, sans-serif',
        'fontSize': '48px',
        'marginBottom': '30px'
    }),

    # Chatbot Description
    html.P(
        "This chatbot uses OpenAI chunking and the Llama 3 embedding model to respond to questions regarding the Fed "
        "Supervision and Regulation (SR) letters. There are 139 SR letters across 5 topics: Liquidity, Market Risks "
        "Management, Banks Security, Credit Risk Management, and Capital Adequacy.",
        style={
            'textAlign': 'center',
            'color': '#34495e',
            'fontSize': '18px',
            'maxWidth': '800px',
            'margin': '0 auto 40px auto',
            'lineHeight': '1.6'
        }
    ),

    # Chatbot Access Button (now placed here with purple color)
    html.Div(
        dcc.Link(
            html.Button("Access Chatbot", style={
                'backgroundColor': '#9b59b6',  # Purple color
                'color': 'white',
                'padding': '14px 32px',
                'borderRadius': '8px',
                'border': 'none',
                'fontSize': '18px',
                'cursor': 'pointer',
                'boxShadow': '0 6px 12px rgba(0,0,0,0.1)',
                'transition': 'all 0.3s ease',
                'marginBottom': '50px'  # Added space below the button
            }),
            href='/chatbot',
            style={'textDecoration': 'none'}
        ),
        style={'display': 'flex', 'justifyContent': 'center'}
    ),

    # Side-by-side layout for Documentation and User Guide
    html.Div([
        # Documentation Section (with link)
        html.Div([
            html.H2("Technical Documentation", style={
                'color': '#1f618d',
                'fontSize': '26px',
                'marginBottom': '10px'
            }),
            html.P("Explore the technical implementation details of our chatbot system.", style={
                'color': '#2c3e50',
                'fontSize': '16px',
                'lineHeight': '1.6',
                'marginBottom': '20px'
            }),
            dcc.Link('View Documentation →', href='/technical-docs', style={
                'color': 'white',
                'textDecoration': 'none',
                'backgroundColor': '#3498db',
                'padding': '10px 20px',
                'borderRadius': '5px',
                'display': 'inline-block'
            })
        ], style={
            'backgroundColor': '#eaf2f8',
            'padding': '25px',
            'borderRadius': '10px',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
            'width': '45%',
            'marginRight': '20px'
        }),

        # User Guide Section (with link)
        html.Div([
            html.H2("User Guide", style={
                'color': '#117a65',
                'fontSize': '26px',
                'marginBottom': '10px'
            }),
            html.P("Learn how to effectively use our chatbot to query SR letters.", style={
                'color': '#2c3e50',
                'fontSize': '16px',
                'lineHeight': '1.6',
                'marginBottom': '20px'
            }),
            dcc.Link('View User Guide →', href='/user-guide', style={
                'color': 'white',
                'textDecoration': 'none',
                'backgroundColor': '#2ecc71',
                'padding': '10px 20px',
                'borderRadius': '5px',
                'display': 'inline-block'
            })
        ], style={
            'backgroundColor': '#e8f8f5',
            'padding': '25px',
            'borderRadius': '10px',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
            'width': '45%'
        }),

    ], style={
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'flex-start',
        'marginBottom': '50px',
        'gap': '20px',
        'flexWrap': 'wrap',
        'maxWidth': '1200px',
        'margin': '0 auto'
    })

], style={
    'backgroundColor': '#f7f9f9',
    'padding': '60px 20px',
    'minHeight': '100vh',
    'fontFamily': 'Inter, sans-serif'
})

# %%
# User Guide Page
user_guide = html.Div([
    html.H1("User Guide", style={
        'color': '#2c3e50',
        'textAlign': 'center',
        'marginBottom': '30px',
    }),
        
   
 html.Div([
        #first text box...
        html.Div([
            html.Div("What can Sophire do?", style={'fontWeight': 'bold', 'marginBottom':'10px','color':'#2c3e50', 'fontSize':'24px','minHeight':'73px'}),
            dcc.Textarea(
                id='textbox1', 
                placeholder='Sophire is a chatbot that can answer questions about policies and procedures found in the Supervision and Regulation (SR) Letters.',
                style={
                    'width':'100%', 
                    'height':'150px', 
                    'fontSize':'18px',
                    'padding':'10px',
                    'boxSizing':'border-box', 
                    'border':'2px solid #cc',
                    'resize':'none',
                    'backgroundColor':'white',
                    'color':'#2c3e50',
                    'fontWeight':'bold'

                }
            )

        ], style={
            'flex':'1', 
            'marginRight':'10px',
            'display':'flex',
            'flexDirection':'column',
            'justifyContent':'flex-start'
        }), 
        #second text box 
        html.Div([
            html.Div("What topics can Sophire answer questions about?", style={'fontWeight':'bold','marginBottom':'10px', 'color':'#2c3e50', 'fontSize':'24px','minHeight':'73px'}),
            dcc.Textarea(
                id='textbox2',
                placeholder='Sophire can answer questions from the following topics of SR letters - Banks Secrecy Act/Office of Foreign Assets Control, Liquidity Risks, Market Risk Management, Capital Adequacy, and Credit Risk.',
                style={
                    'width':'100%', 
                    'height':'150px', 
                    'fontSize':'20px',
                    'padding':'10px',
                    'boxSizing':'border-box', 
                    'border':'2px solid #cc',
                    'resize':'none',
                    'backgroundColor':'white',
                    'color':'#2c3e50',
                    'fontWeight':'bold'
                }

            )
        ], style={
            'flex': '1',
            'display':'flex',
            'flexDirection':'column',
            'justifyContent':'flex-start',
        })

    ], style={
        'display':'flex',
        'width':'100%',
        'padding':'20px',
        'alignItems':'flex-start'
        
    }), 

    html.Div([
        #third text box...
        html.Div([
            html.Div("Does Sophire have time restrictions?", style={'fontWeight': 'bold', 'marginBottom':'10px','color':'#2c3e50', 'fontSize':'24px','minHeight':'73px'}),
            dcc.Textarea(
                id='textbox3', 
                placeholder='This chatbot uses a timeout system, and the timeout for this application is after 15 minutes. This is to make sure the model does not break. There is also a query limit of 30 before the model refreshes itself.',
                style={
                    'width':'100%', 
                    'height':'150px', 
                    'fontSize':'18px',
                    'padding':'10px',
                    'boxSizing':'border-box', 
                    'border':'2px solid #cc',
                    'resize':'none',
                    'backgroundColor':'white',
                    'color':'#2c3e50',
                    'fontWeight':'bold'

                }
            )

        ], style={
            'flex':'1', 
            'marginRight':'10px',
            'display':'flex',
            'flexDirection':'column',
            'justifyContent':'flex-start'
        }), 
        #fourth text box 
        html.Div([
            html.Div("What should you expect from an answer?", style={'fontWeight':'bold','marginBottom':'10px', 'color':'#2c3e50', 'fontSize':'24px','minHeight':'73px'}),
            dcc.Textarea(
                id='textbox4',
                placeholder='After entering your query into the chatbot, you can expect a text answer in around 15 to twenty seconds. This answer will also include a link to the SR letter that Sophire based its answer on.',
                style={
                    'width':'100%', 
                    'height':'150px', 
                    'fontSize':'20px',
                    'padding':'10px',
                    'boxSizing':'border-box', 
                    'border':'2px solid #cc',
                    'resize':'none',
                    'backgroundColor':'white',
                    'color':'#2c3e50',
                    'fontWeight':'bold'
                }

            )
        ], style={
            'flex': '1',
            'display':'flex',
            'flexDirection':'column',
            'justifyContent':'flex-start',
        })

    ], style={
        'display':'flex',
        'width':'100%',
        'padding':'20px',
        'alignItems':'flex-start'
        
    }), 
    #textbox 5...
    html.Div([
        html.Div("What are some best practices when using Sophire?", style={'fontWeight':'bold','marginBottom':'10px', 'color':'#2c3e50', 'fontSize':'24px','minHeight':'73px','textAlign':'center'}),
        dcc.Textarea(
            id='textbox5',
            placeholder='Overall, the quality of Sophire answers improves when the user question is specific and within the scope of topics the model was trained with. Questions that are not relevant or suggest something illegal will make the model provide an erroneous answer.',
            style={
                'width':'100%', 
                'height':'150px', 
                'fontSize':'18px',
                'padding':'10px',
                'boxSizing':'border-box', 
                'border':'2px solid #cc',
                'resize':'none',
                'backgroundColor':'white',
                'color':'#2c3e50',
                'fontWeight':'bold'
            }

        )
    ]),     

  
     
    # Navigation buttons for User Guide
    html.Div([
        dcc.Link(
            html.Button('← Back to Home', style={
                'padding': '10px 20px',
                'marginRight': '20px',
                'backgroundColor': '#5dade2',
                'color': 'white',
                'border': 'none',
                'borderRadius': '6px',
                'cursor': 'pointer',
                'fontSize': '16px'
            }),
            href='/'
        ),
        dcc.Link(
            html.Button('Go to Chatbot →', style={
                'padding': '10px 20px',
                'backgroundColor': '#2ecc71',
                'color': 'white',
                'border': 'none',
                'borderRadius': '6px',
                'cursor': 'pointer',
                'fontSize': '16px'
            }),
            href='/chatbot'
        )
    ], style={
        'textAlign': 'center',
        'margin': '40px auto',
        'maxWidth': '800px'
    })
], style={
    'backgroundColor' : 'white', 
    'color':'#2c3e50', 
    'padding': '20px', 
    'margin' : '0 auto',
})

# Technical Documentation Page
tech_docs = html.Div([
    html.H1("Technical Documentation", style={
        'color': '#2c3e50',
        'textAlign': 'center',
        'marginBottom': '30px'
    }),
    html.Div([
        #sixth text box...
        html.Div([
            html.Div("Background", style={'fontWeight': 'bold', 'marginBottom':'10px','color':'#2c3e50', 'fontSize':'24px','minHeight':'73px'}),
            dcc.Textarea(
                id='textbox6', 
                placeholder='The Sophire chatbot is a Python RAG model trained to answer questions relating to the Federal Reserve Bank’s Supervisory and Regulatory Letters on a corpus of 139 SR letters spanning 5 topics: Market Risk Management, Liquidity, Capital Risks, Capital Adequacy, and Bank Security. Specifically, this chatbot implements breakpoint-based semantic chunking at a 95% threshold using OpenAI’s default text-embedding-ada-002, HuggingFace’s all-MiniLM-L6-v2 sentence transformer model for embeddings, and Meta’s Llama-3.1-8b-instant model for the generated response (Song et al.). Furthermore, we use Groq API as an LPU, which is designed specifically to account for the high computational and memory costs of LLMs (Song et al.).',
                style={
                    'width':'100%', 
                    'height':'150px', 
                    'fontSize':'18px',
                    'padding':'10px',
                    'boxSizing':'border-box', 
                    'border':'2px solid #cc',
                    'resize':'none',
                    'backgroundColor':'white',
                    'color':'#2c3e50',
                    'fontWeight':'bold'

                }
            )

        ], style={
            'flex':'1', 
            'marginRight':'10px',
            'display':'flex',
            'flexDirection':'column',
            'justifyContent':'flex-start'
        }), 
        #seventh text box 
        html.Div([
            html.Div("User Queries", style={'fontWeight':'bold','marginBottom':'10px', 'color':'#2c3e50', 'fontSize':'24px','minHeight':'73px'}),
            dcc.Textarea(
                id='textbox7',
                placeholder='When a user submits a question, the query is split into chunks based on semantic context (semantic chunking) and embedded using the models mentioned before. This is the same process which the SR letters undergo when processing the data for training.The RAG model then searches through the vectors created in the embedding process in the model training step and returns the most relevant content chunk from the database of embeddings generated from the SR letters. The content chunk is passed through the LLM to generate a response to the user in a format that is readable, specific, and includes the corresponding SR letter link thanks to prompt engineering',
                style={
                    'width':'100%', 
                    'height':'150px', 
                    'fontSize':'20px',
                    'padding':'10px',
                    'boxSizing':'border-box', 
                    'border':'2px solid #cc',
                    'resize':'none',
                    'backgroundColor':'white',
                    'color':'#2c3e50',
                    'fontWeight':'bold'
                }

            )
        ], style={
            'flex': '1',
            'display':'flex',
            'flexDirection':'column',
            'justifyContent':'flex-start',
        })

    ], style={
        'display':'flex',
        'width':'100%',
        'padding':'20px',
        'alignItems':'flex-start'
        
    }),
    html.Div([
        #eighth text box...
        html.Div([
            html.Div("Front End", style={'fontWeight': 'bold', 'marginBottom':'10px','color':'#2c3e50', 'fontSize':'24px','minHeight':'73px'}),
            dcc.Textarea(
                id='textbox8', 
                placeholder='The front end application (AKA the part you are using now!) was developed using Dash. SR letters and corresponding attachments were downloaded from the Federal Reserve Bank of Cleveland’s official website and combined into single PDFs for data processing and generating responses. The majority of the SR letters have self-referential links embedded within the file itself, though several do not have any associated links contained within the document. To amend this, links provided in the answers refer to the primary SR letter and were generated through a combination of prompt engineering and web scraping links from the Fed website.',
                style={
                    'width':'100%', 
                    'height':'150px', 
                    'fontSize':'18px',
                    'padding':'10px',
                    'boxSizing':'border-box', 
                    'border':'2px solid #cc',
                    'resize':'none',
                    'backgroundColor':'white',
                    'color':'#2c3e50',
                    'fontWeight':'bold'

                }
            )

        ], style={
            'flex':'1', 
            'marginRight':'10px',
            'display':'flex',
            'flexDirection':'column',
            'justifyContent':'flex-start'
        }), 
        #ninth text box 
        html.Div([
            html.Div("Notes About Citation Links", style={'fontWeight':'bold','marginBottom':'10px', 'color':'#2c3e50', 'fontSize':'24px','minHeight':'73px'}),
            dcc.Textarea(
                id='textbox9',
                placeholder='Due to the inconsistencies in SR letter link formats, some of these links may be inaccurate or missing. In testing on both expert and silly questions, the Sophire chatbot provided the correct response 75 percent off the time and identified the correct SR letter 80 percent of the time. In terms of providing the correct links to SR letters, the chatbot is able to return an accurate link XX percent of the time in testing thus far.',
                style={
                    'width':'100%', 
                    'height':'150px', 
                    'fontSize':'20px',
                    'padding':'10px',
                    'boxSizing':'border-box', 
                    'border':'2px solid #cc',
                    'resize':'none',
                    'backgroundColor':'white',
                    'color':'#2c3e50',
                    'fontWeight':'bold'
                }

            )
        ], style={
            'flex': '1',
            'display':'flex',
            'flexDirection':'column',
            'justifyContent':'flex-start',
        })

    ], style={
        'display':'flex',
        'width':'100%',
        'padding':'20px',
        'alignItems':'flex-start'
        
    }),
    #tenth textbox 
    html.Div([
        html.Div("References", style={'fontWeight':'bold','marginBottom':'10px', 'color':'#2c3e50', 'fontSize':'24px','minHeight':'73px','textAlign':'center'}),
        dcc.Textarea(
            id='textbox5',
            placeholder='Song et. a, “Implementing a RAG Framework for Domain-Specific Data of Federal Reserve Policies,” 2014',
            style={
                'width':'100%', 
                'height':'150px', 
                'fontSize':'18px',
                'padding':'10px',
                'boxSizing':'border-box', 
                'border':'2px solid #cc',
                'resize':'none',
                'backgroundColor':'white',
                'color':'#2c3e50',
                'fontWeight':'bold'
            }

        )
    ]),

    
    # Navigation buttons for Technical Docs
    html.Div([
        dcc.Link(
            html.Button('← Back to Home', style={
                'padding': '10px 20px',
                'marginRight': '20px',
                'backgroundColor': '#5dade2',
                'color': 'white',
                'border': 'none',
                'borderRadius': '6px',
                'cursor': 'pointer',
                'fontSize': '16px'
            }),
            href='/'
        ),
        dcc.Link(
            html.Button('Go to Chatbot →', style={
                'padding': '10px 20px',
                'backgroundColor': '#2ecc71',
                'color': 'white',
                'border': 'none',
                'borderRadius': '6px',
                'cursor': 'pointer',
                'fontSize': '16px'
            }),
            href='/chatbot'
        )
    ])
], style={
    'backgroundColor' : 'white', 
    'color' : '#2c3e50'
})

# %%

#Layout with Page Routing. Default to landing page when running the app
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),  # Page navigation
    html.Div(id="page-content"),  # Page content will change dynamically
    dcc.Store(id='session-id', data=str(uuid.uuid4())),  # Unique session ID
    html.Button('Send', id='submit-btn',style={
    'display': 'none'  # Removes from layout entirely
}), # Avoid call back errors
    html.Button('Refresh', id='refresh-btn', style={
    'display': 'none'  # Removes from layout entirely
}),
    html.Div(id='chat-window') # Avoid call back errors
    
])

chatbot_page = html.Div([
    # Title
    html.H1("Sophire Chatbot", style={
        'textAlign': 'center',
        'color': '#2c3e50',
        'fontFamily': 'Inter, sans-serif',
        'fontSize': '36px',
        'marginBottom': '30px'
    }),

    # Chat window
    html.Div(id="chat-window", style={
        'height': '500px',
        'overflowY': 'auto',
        'padding': '20px',
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
        'marginBottom': '20px'
    }),

    # Input area
    html.Div([
        dcc.Input(
            id="query-input",
            type="text",
            placeholder="Ask a question about SR letters...",
            style={
                'flex': 1,
                'padding': '12px 20px',
                'borderRadius': '8px',
                'border': '1px solid #ddd',
                'fontSize': '16px'
            }
        ),
        html.Button(
            "Send",
            id="submit-btn",
            n_clicks=0,
            style={
                'backgroundColor': '#5dade2',
                'color': 'white',
                'padding': '12px 24px',
                'borderRadius': '8px',
                'border': 'none',
                'marginLeft': '10px',
                'fontSize': '16px',
                'cursor': 'pointer'
            }
        )
    ], style={'display': 'flex', 'marginBottom': '30px'}),
    
    # Back button
    html.Div(
        dcc.Link(
            html.Button('← Back to Home', style={
                'padding': '12px 24px',
                'backgroundColor': '#5dade2',
                'color': 'white',
                'border': 'none',
                'borderRadius': '8px',
                'cursor': 'pointer',
                'fontSize': '16px'
            }),
            href='/'
        ),
        style={'textAlign': 'center'}
    ), 
    
    html.Button('⟳ New Session', id='refresh-btn', style={
    'marginLeft': '10px',
    'background': '#ff6b6b',
    'color': 'white',
    'borderRadius': '5px',
    'padding': '10px 15px'
})

], style={
    'backgroundColor': '#f7f9f9',
    'padding': '40px 20px',
    'minHeight': '100vh',
    'fontFamily': 'Inter, sans-serif',
    'maxWidth': '800px',
    'margin': '0 auto'
})

# %%
from dash import callback_context
from dash import no_update
# URL Routing to navigate between the pages
@app.callback(Output("page-content", "children"),
              Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/user-guide":
        return user_guide
    elif pathname == "/technical-docs":
        return tech_docs
    elif pathname == "/chatbot":
        return chatbot_page
    else:
        return landing_page
    
# Consolidated chat callback that handles both formatting and processing
@app.callback(
    [Output('chat-window', 'children'),
     Output('query-input', 'value'),
     Output('session-id', 'data')],
    [Input('submit-btn', 'n_clicks'),
     Input('query-input', 'n_submit'),
     Input('refresh-btn', 'n_clicks')],
    [State('session-id', 'data'),
     State('query-input', 'value'),
     State('chat-window', 'children')],
    prevent_initial_call=True
)

def handle_all_actions(submit_clicks, enter_presses, refresh_clicks, session_id, user_input, chat_history):
    ctx = callback_context
    
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    # Check what buttons the user clicked (refresh/ submit/ or enter query)
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle Refresh action
    if trigger_id == 'refresh-btn':
        new_session_id = str(uuid.uuid4())
        return [], "", new_session_id  # Clear chat, clear input, new session
    
    # Handle Submit/Enter actions
    elif trigger_id in ('submit-btn', 'query-input'):
        if not user_input:
            return no_update, no_update, no_update
            
        # Initialize or validate session
        now = datetime.now()
        if session_id not in sessions:
            sessions[session_id] = {
                'start_time': now,
                'last_active': now,
                'query_count': 0
            }
        
        # Check timeout (15 minutes)
        if (now - sessions[session_id]['last_active']) > timedelta(minutes=15):
            timeout_msg = html.Div("Session expired. Please refresh.",
                                 style={'color': 'red', 'textAlign': 'center'})
            return (chat_history or []) + [timeout_msg], "", no_update
        
        # Check query limit (30 queries)
        if sessions[session_id]['query_count'] >= 30:
            limit_msg = html.Div("Query limit reached. Please refresh.",
                               style={'color': 'red', 'textAlign': 'center'})
            return (chat_history or []) + [limit_msg], "", no_update
        
        # Update session
        sessions[session_id]['last_active'] = now
        sessions[session_id]['query_count'] += 1
        
        # Process valid message
        user_msg = html.Div(
            html.Div(
                user_input,
                style={
                    'display': 'inline-block',
                    'padding': '12px 16px',
                    'backgroundColor': '#0055A2',
                    'color': 'white',
                    'borderRadius': '18px 18px 0 18px',
                    'maxWidth': '80%'
                }
            ),
            style={'textAlign': 'right', 'margin': '8px 0'}
        )
        
        response, _ = pipe.generate(user_input)
        #response = query_engine.query(user_input).response #Uncomment this line if you want to fix the interface without changing the model
        
        bot_response = html.Div(
    dcc.Markdown(
        response,
        style={
            'whiteSpace': 'pre-wrap',
            'color': 'white',
            'backgroundColor': '#5A5A5A',
            'padding': '12px 16px',
            'borderRadius': '18px 18px 18px 0',
            'maxWidth': '80%',
            'display': 'inline-block',
        },
        dangerously_allow_html=False  # Keep this False if you're using Markdown
    ),
    style={'textAlign': 'left', 'margin': '8px 0'}
)
        
        return (chat_history or []) + [user_msg, bot_response], "", no_update
    
    # Handle if the user does not click anything (no trigger id)
    return no_update, no_update, no_update

# Run the app locally
if __name__ == "__main__":
    app.run(port=8050,debug=True, mode='browser')
    
    # Comment this out if you want to test
    #csv_test("quarto-manuscript/questions/new questions - april 2025/Expert All columns.csv", pipe)
    #csv_test("quarto-manuscript/questions/new questions - april 2025/New Expert Questions.csv", pipe)

    # FOR MANUAL TESTING
    # irrelevant_test_questions = [
    #     "What is the meaning of life"
    # ]
    
    # for q in irrelevant_test_questions:
    #     print(f"\nQUESTION: {q}")
    #     try:
    #         answer, sources = pipe.generate(q)
    #         print(f"Answer: {answer}")
    #         # print(f"SOURCES:\n{sources}")
    #     except Exception as e:
    #         print(f"Error while answering question: {e}")







