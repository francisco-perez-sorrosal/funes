import importlib.util
import multiprocessing
import os
import pathlib

from typing import Optional, Sequence

import dspy
import streamlit as st
import pandas as pd

from dspy.datasets.hotpotqa import HotPotQA
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from funes.interactions import FactualSortQA
from funes.lang_utils import models, get_llm


st.set_page_config(layout="wide")


MODEL_BASEDIR=pathlib.Path(os.environ.get("HOME", ""), "models", "funes", "react")
st.write(f"Model base directory: {MODEL_BASEDIR}")

num_cpus = multiprocessing.cpu_count()
st.write(f"Number of cores: {num_cpus}")

def check_package_availability(package_name):
    spec = importlib.util.find_spec(package_name)
    return spec is not None

if check_package_availability("vertexai"):
    import vertexai
   
    PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
    LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    st.write(f"The Vertex AI package is installed. Using project {PROJECT_ID} in region {LOCATION}.")
else:
    print("The numpy package is not installed.")


def generate_response(program, input_text):
    with st.spinner("Generating response..."):
        return program(question=input_text)
    
def display_response(response):
    col1, col2 = st.columns(2)
    with col1:
        st.info(response.answer)
    with col2:
        if isinstance(response, dspy.Prediction):
            for obs in response.observations:
                st.warning(obs)
        else:
            st.info(response.rationale)


@st.cache_data
def get_dataset_splits(train_size=200, dev_size=300, test_size=10, train_seed=1, eval_seed=2023):
    # HotPotQA doesn't have a test set so...
    initial_train_size = train_size + dev_size
    dataset = HotPotQA(train_seed=train_seed, train_size=initial_train_size, eval_seed=eval_seed, dev_size=test_size, test_size=0)
    trainset = [x.with_inputs('question') for x in dataset.train[0:train_size]]
    valset = [x.with_inputs('question') for x in dataset.train[train_size:initial_train_size]]
    testset = [x.with_inputs('question') for x in dataset.dev[0:test_size]]
    return trainset, valset, testset


def show_inspect(llm):
    history = llm.inspect_history(n=20)
    chunked_hist = history.split("\n\n\n")
    with st.expander("Text List"):
        for text in chunked_hist:
            if text != "":
                with st.chat_message("assistant"):
                    st.text(text)

def show_qa_example(split, index, prediction: Optional[str]=None, scores_bool: Optional[Sequence[bool]]=None):
    q = split[index]['question']
    a = split[index]['answer']
    
    resp = f"\n#### Q: {q}"
    if prediction is not None:
        if prediction == '':
            prediction = 'N/A'
    resp += f"\n#### GT/Predicted Answer: :green[{a}] / :blue[{prediction}]"

    if (scores_bool is not None and scores_bool[index]) or (a == prediction):
        st.success(resp)
    else:
        st.error(resp)
        
def main():
    st.title("DSPy App with Vertex AI!")

    # Dropdown box for selecting options
    model_options = list(models.keys())
    selected_option = st.selectbox("Select an option", model_options, index=len(model_options)-1)
    
    model = models.get(selected_option, "")
    if model == "":
        st.warning("No model selected")
        st.stop() 
    st.write(model)
    llm = get_llm(model, "TGI", port=8081)
    
    # st.write(llm.url)
    # st.write(llm.ports)
    # st.json(llm.http_request_kwargs)
    # st.json(llm.headers)
    # st.json(llm.kwargs)
    
    # example_col, train_col = st.columns(2)
       
    # with example_col:
    #     dspy.configure(lm=llm)
    #     with st.form('my_form'):
    #         text = st.text_area('Enter text:', 'What is the capital of Paris?')
    #         submitted = st.form_submit_button('Submit')
    #         if submitted:
    #             qa = dspy.ChainOfThought('question -> answer')
    #             response = generate_response(qa, text)
    #             display_response(response)
        
    # with train_col:
    colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    dspy.configure(lm=llm, rm=colbert)
    
    with st.popover("Dataset Sizes"):
        st.markdown("Split Sizes -> Train: 300, Dev: 100, Test: 10")
        train_size = int(st.text_input("Train Size", value="300"))
        val_size = int(st.text_input("Val Size", value="100"))
        test_size = int(st.text_input("Test Size", value="10"))
        
    
    trainset, valset, testset = get_dataset_splits(train_size=train_size, dev_size=val_size, test_size=test_size)
    st.markdown(f"### Dataset info: Train: {len(trainset)}, Val: {len(valset)}, Dev: {len(testset)}")

    valset_col, testset_col = st.columns(2)
    with valset_col:
        st.markdown(f"#### Example Val Set")
        val_df = pd.DataFrame([x.toDict() for x in valset])
        st.dataframe(val_df)

    with testset_col:
        st.markdown(f"#### Example Test Set")
        test_df = pd.DataFrame([x.toDict() for x in testset])
        st.dataframe(test_df)

        # Add a multiselect widget to select rows based on the index
        selected_indices = st.multiselect('Select rows:', test_df.index, max_selections=1, default=[0])
        
    st.divider()
    
    st.markdown("### Train and Optimize a ReAct Model!!!")

    agent_creation_col, agent_load_col = st.columns(2)
    with agent_creation_col:
        clear_agent_btn = st.button("Clear Agent Session")
        if clear_agent_btn:
            del st.session_state["agent"]
        if 'agent' in st.session_state:
            agent = st.session_state['agent']
            st.text("Using agent from session")
        else:
            agent = dspy.ReAct(FactualSortQA, tools=[dspy.Retrieve(k=1)])
            st.session_state['agent'] = agent
            st.text("New agent created")

    with agent_load_col:
        model_files = [file.name for file in MODEL_BASEDIR.glob("*.json")]
        selected_model = st.selectbox("Select a model", model_files)
        if selected_model:
            load_filename = str(MODEL_BASEDIR / selected_model)

        load_chk = st.checkbox('Load Agent', value=False)
        if load_chk and agent:
            st.text("Loading agent...")
            agent.load(load_filename)
    
    with st.expander("ReAct Model Description"):
        st.write(agent)
    
    example_question = testset[selected_indices[0]]['question']

    opt_col, save_agent_col = st.columns(2)
    with opt_col:
        with st.form('opt_form'):
            opt_dev_eval_submitted = st.form_submit_button('Optimize Eval')
            if opt_dev_eval_submitted:
                with st.spinner("Optimizing..."):
                    config = dict(max_bootstrapped_demos=2, max_labeled_demos=0, num_candidate_programs=5, num_threads=2*num_cpus)
                    tp = BootstrapFewShotWithRandomSearch(metric=dspy.evaluate.answer_exact_match, **config)
                    opt_agent = tp.compile(agent, trainset=trainset, valset=valset)
                    st.text("Adding optimized agent to session")
                    st.session_state['opt_agent'] = opt_agent

    with save_agent_col:
        with st.form('save_agent_form'):
            save_filename = st.text_input('Enter filename:', value=f'{MODEL_BASEDIR}/optimized_react_{model.split("/")[-1]}_{train_size}_train_examples.json')
            save_btn = st.form_submit_button('Save Optimized Agent')
            if save_btn and st.session_state.get('opt_agent'):
                st.text("Saving optimized agent")                
                opt_agent=st.session_state.get('opt_agent')
                opt_agent.save(save_filename)
            else:
                st.warning("No optimized agent in session to save")
            

    with st.form('validate'):
        use_optimized = st.checkbox('Use Optimized Agent', value=False)
        if use_optimized:
            if st.session_state.get('opt_agent'):
                st.info("Using optimized agent from session")
                eval_agent = st.session_state['opt_agent']
            else:
                st.warning("No optimized agent in session. Using default agent...")
                eval_agent = agent
        else:
            eval_agent = agent
        config = dict(num_threads=2*num_cpus, display_progress=True, display_table=5)
        validation_submitted = st.form_submit_button('Validate')
        if validation_submitted:
            with st.spinner("Evaluating..."):
                evaluate = Evaluate(devset=testset, metric=dspy.evaluate.answer_exact_match, **config)
                score, outputs, scores_bool = evaluate(eval_agent, return_all_scores = True, return_outputs = True)
                st.markdown(f"### Evaluation Score: {int(score)}/{len(testset)}")
                st.markdown(f"## Scores")
                for i, s in enumerate(outputs):
                    example, prediction, score = outputs[i]
                    st.write(prediction)
                    assert testset[i]['question'] == example['question']
                    show_qa_example(testset, i, str(prediction['answer']), scores_bool)
                    st.json(example.toDict(), expanded=False)

                                    
    with st.form('test_example_form'):
        text = st.text_area('Enter text:', example_question)
        opt_example_submitted = st.form_submit_button('Submit Example')
        if opt_example_submitted:
            response = generate_response(agent, text)
            display_response(response)            

    show_inspect(llm)

if __name__ == "__main__":
    main()
