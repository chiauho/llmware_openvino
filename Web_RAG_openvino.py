# https://github.com/llmware-ai/llmware/blob/main/fast_start/rag/example-3-prompts_and_models.py
# A couple of issues to note:
# Need to pip install openvino openvino-genai
# After this, may still have error with DLL dependencies. So need to resolve this.
# pip install --upgrade openvino-dev
# Microsoft Visual C++ Redistributable is also required
# Remember to keep numpy at version 1.26.4.

# We will be using llmware quantized models specially tuned for openvino

import time
from llmware.prompts import Prompt
from llmware.models import ModelCatalog
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pdf2image
import os
import fitz  # must install PyMuPDF

from langchain_chroma import Chroma     # replace from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings     # replace from langchain_community.embeddings import HuggingFaceEmbeddings
# Install numpy == 1.26.4. do not upgrade. higher version does not work with some of the libraries.


# Here we initialize the FastAPI application and set up Jinja2 for template rendering
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="/users/chiauho.ong/LLM/llmware_openvino/static"), name="static")


# This defines a Pydantic model for validating user input.
class UserInput(BaseModel):
    user_input: str


def generate_images(file_name_list, page_list):
    images = []
    pdf_path = "\\users\chiauho.ong\\LLM\\llmware_openvino"     # full path
    img_path = "\\users\\chiauho.ong\\LLM\\llmware_openvino\\static"   # full path
    for pdf_file, page_num in zip(file_name_list, page_list):
        basename = os.path.basename(pdf_file)
        read_pdf_file = f"{pdf_path}\\{pdf_file}"  # Full path to PDF file
        if os.path.exists(read_pdf_file):
            pages = pdf2image.convert_from_path(read_pdf_file, first_page=page_num, last_page=page_num)
            if pages:
                img_file_name = f"{img_path}\\{basename}_page{page_num}.jpg"
                pages[0].save(img_file_name, 'JPEG')
                images.append(f"\\static\\{basename}_page{page_num}.jpg")  # Use relative path for client
    return images


# This route handler serves the main HTML page.
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process_input")
async def process_input(user_input: UserInput):
    output_text, file_name_list, page_list = submit_query(user_input.user_input)
    images = generate_images(file_name_list, page_list)
    return JSONResponse(content={"output_text": output_text, "images": images})


def order_text_by_numbers(list_of_text, list_of_num, k):
    # Create a list of tuples, where each tuple contains the number and its corresponding text
    paired_list = list(zip(list_of_num, list_of_text))

    # Sort the paired list based on the numbers in descending order
    sorted_pairs = sorted(paired_list, key=lambda x: x[0], reverse=True)

    # Extract only the text from the sorted pairs
    list_of_ordered_text = [pair[1] for pair in sorted_pairs[:k]]

    return list_of_ordered_text
# end function


def get_unique_texts(text_list):
    seen = set()
    unique_texts = []
    for text in text_list:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    return unique_texts


def get_unique_elements(pages, filenames):
    seen = set()
    pages_unique = []
    filenames_unique = []

    for page, filename in zip(pages, filenames):
        combined = f"{page}{filename}"
        if combined not in seen:
            seen.add(combined)
            pages_unique.append(page)
            filenames_unique.append(filename)

    return pages_unique, filenames_unique


def get_embedding_function(e_model, m_kwargs, e_kwargs):
    # returns an embedding function from huggingface
    e_function = HuggingFaceEmbeddings(
        model_name=e_model,
        model_kwargs=m_kwargs,
        encode_kwargs=e_kwargs)
    return e_function
# end function


def read_from_chromadb(cdir, e_model, m_kwargs, e_kwargs):
    e_function = get_embedding_function(e_model, m_kwargs, e_kwargs)
    cdb = Chroma(persist_directory=cdir, embedding_function=e_function)
    return cdb
# End of function


def compare_retrieve(u_query, dbase, top_k):
    # Make a retriever to return top_k results
    # retriever = dbase.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8, "k": top_k})
    retriever = dbase.as_retriever(search_kwargs={"k": top_k})
    results = retriever.invoke(u_query)
    # [d[1] for d in dbase.similarity_search_with_relevance_scores(u_query, k=5)]

    # Use reranker. The reranker will return from top_k the most relevant results
    # rerank = use_reranker(results)
    return results
# End of function


def affirmative_prompt(u_query, model):
    # First do the rephrase
    query = f"""You are a translator. Your job is to translate a question into an affirmative statement. For example:
Question: What is my required notice period when I quit the company?
Answer: When I quit the company, I need to give a notice period of 

It is very important that you only answer with the affirmative statement and nothing else. Do not ask if there is anything else I would like to translate or say anything other
than the required affirmative text.
Question: {u_query}
Answer:"""
    response = model.inference(query)
    affirmative_text = response['llm_response']
    print(f"Affirmative text: {affirmative_text}\n")
    return affirmative_text
# end of function


def prompt_formatter(u_query, context_items, tokenizer):
    # Augment query with text-based context from context_items
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join(item for item in context_items)
    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = f"""Using the information contained in the context, give a comprehensive answer to the question.
Make sure your answers are as explanatory as possible. Always state the sections and pages where you found your answers.
Use the following examples as reference for the ideal answer style.
Example 1:
Question: I am a senior manager. I have just resigned. What is my required notice period?
Answer: If you are a senior manager and if you have just resigned, your notice period is xx months based on the information found on section yy on page zz.

Example 2:
Query: What are the type of benefits I can expect from the company?
Answer: Based on the information in section aa on page 22 and section bb on page 10 and section cc on page 100, here are the various types of benefits the company provides:
Medical Leave: Your extracted summary here
Compassionate Leave: Your extracted summary here
And other benefits.

Now use the following context items to answer the question:
{context}

Question: {u_query}
Answer:"""

    # Create prompt template for instruction-tuned model
    # Different models expect different format
    # That's why we use tokenizer.apply_chat_template() so that we don't have to figure out
    # what is the format
    dialogue_template = [
        {"role": "user",
         "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt
# End of function


def llm(context_items, u_query, filename_list, page_list):
    global g_model

    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join(item for item in context_items)
    start_time = time.perf_counter()
    response = g_model.inference(u_query, add_context=context)
    end_time = time.perf_counter()
    lapse = (end_time-start_time) / 60
    print(f"It took this number of minutes: {lapse}\n")

    output_text = response['llm_response']

    adjusted_page_list = [i + 1 for i in page_list]  # adjust page number
    basename_list = [os.path.basename(filename) for filename in filename_list]
    output_text_with_fileinfo = str(list(zip(basename_list, adjusted_page_list)))
    output_text_with_fileinfo = "Filename & Page info are " + output_text_with_fileinfo + "\n\n" + output_text

    print(f"Query: {u_query}")
    print(f"RAG answer:\n{output_text_with_fileinfo}")

    # Remove prompt, <bos>, <eos>
    # output_text_cleaned = output_text_with_fileinfo.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')
    return output_text_with_fileinfo
# End of function


def submit_query(user_input: str):
    global g_user_query, g_db, g_embedding_model, g_topk, g_llm, g_model
    g_user_query = user_input
    print("User Query:", g_user_query)  # Stored user input passed from web UI

    # First turn the question into an affirmative statement. This makes it closer in semantic to the stored text. Should result in better search.
    # a_prompt = affirmative_prompt(g_user_query, g_model)
    a_prompt = g_user_query

    # Returns the top k results in langchain Document format
    results = compare_retrieve(a_prompt, g_db, g_topk)
    filename_list = []
    page_list = []
    text_list = []
    for doc in results:
        filename_list.append(doc.metadata['source'])
        page_list.append(doc.metadata['page'])  # note page number is zero index. so need to +1 later to get the correct page
        # text_list.append(doc.page_content)

    # Because sometimes duplicate is returned. I want only unique page.
    # However, text_list can be showing same page but different section of the text, depending on
    # the embedding.
    page_list, filename_list = get_unique_elements(page_list, filename_list)
    # text_list = get_unique_texts(text_list)
    for page, filename in zip(page_list, filename_list):
        doc = fitz.open(filename)
        p = doc.load_page(page)
        text_list.append(p.get_text())

    # Display the pdf pages
    print(f"Filename list = {filename_list}\n")
    print(f"Page list (start with page 0, not 1) = {page_list}\n")

    adjusted_page_list = [i+1 for i in page_list]     # adjust page number
    print(f"Call llm ... \n")
    output_text_llm = llm(text_list, g_user_query, filename_list, page_list)
    return output_text_llm, filename_list, adjusted_page_list   # page adjusted because later the pdf2image function is base 1, not base 0.
# end function


def delete_files_in_static_dir():
    directory = './static'
    if not os.path.exists(directory):
        return 1    # No directory static
    files = os.listdir(directory)
    if not files:
        return 2    # No files in directory static
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):   # make sure it's a file, not a subdirectory
            os.remove(file_path)
    return 0
# end of function


if __name__ == "__main__":
    import uvicorn
    global g_db, g_user_query, g_embedding_model, g_tokenizer
    global g_topk, g_llm, g_model

    g_topk = 5  # retrieve top x documents match to query. Later use rerank to limit to 5 documents for display
    g_embedding_model = "sentence-transformers/all-mpnet-base-v2"  # "hkunlp/instructor-base"
    g_llm = "llama-3.2-3b-instruct-ov"
    # chromadb_dir = "./chroma_all-mpnet-base-v2-hr_docs"     # where to retrieve stored text and embeddings
    chromadb_dir = "./chroma_test"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    # First clean up static directory. Delete all jpg files.
    code = delete_files_in_static_dir()
    if code == 0:
        print("All files in directory static deleted\n")
    elif code == 1:
        print("No directory name static found\n")
    else:
        print("No files in directory static found\n")

    # Load embeddings for 1st time. No need to load subsequently.
    print(f"Load embeddings ....\n")
    g_db = read_from_chromadb(chromadb_dir, g_embedding_model, model_kwargs, encode_kwargs)

    # Load the model upon startup
    g_model = ModelCatalog().load_model(g_llm, temperature=0.0, sample=False, max_output=1000)

    # Run Fast API
    uvicorn.run(app, host="0.0.0.0", port=80)
