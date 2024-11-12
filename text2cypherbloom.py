import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import sys
sys.path.append("./_drivers")
from _drivers.llama3qahelper import custom_llama_qahelper
myBloomlist= ["bank account transfer", "disease path", "financial investment", "person interacted with websites","person-company-financial product"]

st.set_page_config(page_title="Free question helper",layout="wide")
st.title("Fraud Detection - from natural language to Bloom visualization")
image = Image.open('visualisation.png')
st.image(image, width=800)
st.title("Free Text Area Input Form")
st.markdown("""
<style>
table {
    width: 100%;
    border-collapse: collapse;
    border: none !important;
    font-family: "Source Sans Pro", sans-serif;
    color: rgba(49, 51, 63, 0.6);
    font-size: 0.9rem;
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
    <th colspan="3">Sample Questions</th>
</tr>
<tr>
    <td>Identify clients sharing PII</td>
    <td>Client Henry Miranda performed how many transactions?</td>
    <td>FirstPartyFraudster with firstPartyFraudScore greater than 2.9 return his name, email and phone</td>
</tr>
<tr>
    <td>Client Sydney Jacobson shared SSN with another Client, return the entire path</td>  
    <td>Client Naomi Rodriguez to Merchant- MErntogra Inc. return the entire path</td>
    <td>FirstPartyFraudster transfer to other Clients, return the entire path</td>  
    <td>Clients have Transaction with FirstPartyFraudster return the entire path</td>               
</tr>
</table>
""", unsafe_allow_html=True)

with st.form(key='my_form'):
    text_input = st.text_area("Enter your question here:")
    submit_button = st.form_submit_button("Submit")

if submit_button:
    st.write("You submitted:")
    st.write(text_input)
    my_question = text_input 
    valid_cypher,valid_context = custom_llama_qahelper(my_question)
    st.write(valid_cypher)
    st.write(valid_context)

    actual_phrase =  "runCypher" + str(valid_cypher).replace(" ","%20").replace(";","")
    iframe_src =f"""https://bloom.neo4j.io/index.html?connectURL=ab7a7fae.databases.neo4j.io&run=true&search={actual_phrase}"""
    components.iframe(iframe_src, height=800, scrolling=True)

