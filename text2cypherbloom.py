import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import pandas as pd
import sys
sys.path.append("./_drivers")
from _drivers.llama3qahelper import custom_llama_qahelper
from _drivers.neo4j_handler import write_cypher_query,exec_cypher_query
# from _drivers.apitokenhelper import fetchToken

myBloomlist= ["bank account transfer", "disease path", "financial investment", "person interacted with websites","person-company-financial product"]

st.set_page_config(page_title="Free question helper",layout="wide")
st.title("Fraud Detection - Bot & Plot")
image = Image.open('visualisation.png')
st.image(image,width=1000)
st.title("Input your question to generate Cypher (Bot) and then visualize results in Bloom (Plot)")
st.markdown("""
<style>
table {
    width: 100%;
    border-collapse: collapse;
    border: none !important;
    font-family: "Source Sans Pro", sans-serif;
    color: rgba(49, 51, 63, 0.8);
    font-size: 1.3rem;
}

tr {
    border: none !important;
}

th {
    text-align: center;
    colspan: 3;
    border: none !important;
    color: #0F9D58;
}

th, td {
    padding: 2px;
    border: none !important;
}
</style>

<table>
<tr>
    <th colspan="5">Some Sample Questions</th>
</tr>
<tr>
    <td>Client Henry Miranda performed how many transactions?</td>
    <td>FirstPartyFraudster with firstPartyFraudScore greater than 2.9 return his name, email and phone</td>
</tr>
<tr>
    <td>Client Sydney Jacobson shared SSN with another Client, return the entire path</td>  
    <td>Client Naomi Rodriguez to Merchant- MErntogra Inc. return the entire path</td>
</tr>
<tr>
    <td>FirstPartyFraudster transfer to other Clients, return the entire path</td>  
    <td>Clients have Transaction with FirstPartyFraudster return the entire path</td>             
</tr>
<tr>
    <td>Identify clients sharing PII</td>
    <td>Circular money transfer from Bank to Bank, up to 3 hops, return the entire path</td>             
</tr>
</table>
""", unsafe_allow_html=True)

def fetch_options():
    queryLLMCypher = """
    MATCH (n:LLMCypher)
    RETURN n.key AS key, n.value AS value
    """
    result = exec_cypher_query(queryLLMCypher)

    df = pd.DataFrame(result)
    return df

with st.form(key='my_form',clear_on_submit=False):
    text_input = st.text_area("Enter your question here:")
    submit_button = st.form_submit_button("Submit", type="primary")
    submit_likeButtion = st.form_submit_button("Like & Save")

if submit_button:
    st.write("You submitted:")
    st.write(text_input)
    my_question = text_input 
    valid_cypher,valid_context = custom_llama_qahelper(my_question)
    st.write(valid_cypher)
    
    st.session_state['liked'] = valid_cypher

    actual_phrase =  "runCypher" + str(valid_cypher).replace('\n',' ').replace(" ","%20").replace(";","")
    # myAuraToken = fetchToken()
    # print(myAuraToken)
    # iframe_src =f"""https://bloom.neo4j.io/index.html?connectURL=ab7a7fae.databases.neo4j.io&run=true&authToken={myAuraToken}&search={actual_phrase}"""
    # iframe_src =f"""https://bloom.neo4j.io/index.html?connectURL=ab7a7fae.databases.neo4j.io&run=true&search={actual_phrase}"""
    iframe_src = st.secrets["NEO4J_BLOOM_URL"] + actual_phrase

    components.iframe(iframe_src, height=800, scrolling=True)

    st.write(valid_context)

if submit_likeButtion:
    if 'liked' not in st.session_state:
        st.warning("You need to input a question to get Cypher first")
    else:
        cypher_result = st.session_state['liked'].replace('\n',' ')
 
        st.info("you will save Cypher into db")
        storeCypher_str =f"""
        merge (n:LLMCypher {{key:'{text_input}'}}) set n.value ='{cypher_result}'
        """

        store_result = write_cypher_query(storeCypher_str)

        st.info(storeCypher_str)

with st.form(key='Bloom_Form',clear_on_submit=False):

    queryLLMCypher = """
    MATCH (n:LLMCypher)
    RETURN n.key AS key, n.value AS value
    """

    queryLLMCypherResult = exec_cypher_query(queryLLMCypher)
    df = pd.DataFrame(queryLLMCypherResult,columns=["key", "value"])

    # Check if DataFrame is empty
    if df.empty:
        options =("empty")
    else:
        options = df.apply(lambda row: f"{row['key']} ##### Stored in AuraDB as #####=> {row['value']}", axis=1).tolist()

        selected_option = st.selectbox("Select a phrase to search Bloom", options)
        # st.write("You selected:", selected_option)

        searchPhraseBloom = selected_option.split("=>", 1)[1].strip()
        submit_displayBloom = st.form_submit_button("Display Bloom",type="primary")

        if submit_displayBloom:
            actual_phrase =  "runCypher" + str(searchPhraseBloom).replace(" ","%20").replace(";","")
            # iframe_src =f"""https://bloom.neo4j.io/index.html?connectURL=ab7a7fae.databases.neo4j.io&run=true&search={actual_phrase}"""
            iframe_src = st.secrets["NEO4J_BLOOM_URL"] + actual_phrase
            components.iframe(iframe_src, height=800, scrolling=True)



