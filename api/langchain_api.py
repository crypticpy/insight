from langchain.callbacks import LangChainTracer
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain_openai import ChatOpenAI
from config import config
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "streamlit-rag-app"

tracer = LangChainTracer(project_name="streamlit-rag-app")

def run_evaluation(qa_chain):
    eval_config = RunEvalConfig(
        evaluators=[
            "qa",
            "context_relevancy",
            "answer_relevancy",
        ],
        custom_evaluators=[],
        eval_llm=ChatOpenAI(model=config.OPENAI_MODEL_NAME, temperature=0),
    )

    eval_results = run_on_dataset(
        client=tracer.client,
        dataset_name="your_dataset_name",
        llm_or_chain_factory=lambda: qa_chain,
        evaluation=eval_config,
    )

    return eval_results
