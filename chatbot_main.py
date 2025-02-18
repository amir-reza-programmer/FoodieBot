import chainlit as cl
import json
import re
from tavily import TavilyClient
from helperr import get_order_status, cancel_order, comment_order, search_food_orders, load_key
from mylancedb import answering_general_questions, DocumentEmbedding
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(api_key=load_key(
    "GEMINI_API_KEY"), model="gemini-1.5-flash")


def call_api(usere_message, system_prompt):
    messages = [
        ("system", system_prompt),
        ("user", usere_message),
    ]
    response = llm.invoke(messages)
    return response.content


def handle_intend(message):
    system_prompt = "find the intend of the message and return these values: CHECK_ORDER, CREATE_COMMENT, CANCELL_ORDER, SEARCH_FOOD or GENERAL_QUESTION"
    response = call_api(message, system_prompt)
    return response


def handel_check_order(message):
    system_prompt = 'extract just the order id from the user message it is any number, if there was no order id return None'
    response = call_api(message, system_prompt)
    while True:
        print(response)
        if not 'none' in response.lower():
            cl.user_session.set("current_state", None)
            cl.user_session.set("chat_history", "")
            return get_order_status(int(response))
        else:
            message = 'enter your order id:'
            return message


def handel_cancel_order(message):
    system_prompt = "extract from user message telephone number and order id if not exist each of them set None; response in form -> ORDER_ID:VALUE,PHONE_NUM:VALUE "
    response = call_api(system_prompt, message)

    order_id_match = re.search(r"ORDER_ID:(\d+)", response)
    phone_num_match = re.search(r"PHONE_NUM:([\d\-]+)", response)

    order_id = order_id_match.group(1) if order_id_match else None
    phone_num = phone_num_match.group(1) if phone_num_match else None
    print("ORDER_ID:", order_id)
    print("PHONE_NUM:", phone_num)

    if order_id is None:
        return 'enter your order id'
    elif phone_num is None:
        return 'enter your phone number'

    else:
        result = cancel_order(int(order_id), phone_num)
        if result:
            cl.user_session.set("current_state", None)
            cl.user_session.set("chat_history", "")
            return "Your Order Successfuly Cancelled!"
        else:
            cl.user_session.set("chat_history", "")
            return "there is a problem, try again"


def handle_comment(message):
    system_prompt = "extract from user message telephon number(in format of xxx-xxx-xxxx) and order id and comment if not exist each of them set None; response in form -> ORDER_ID:VALUE,PHONE_NUM:VALUE,COMMENT:VALUE"
    response = call_api(system_prompt, message)

    order_id_match = re.search(r"ORDER_ID:(\d+)", response)
    phone_num_match = re.search(r"PHONE_NUM:([\d\-]+)", response)
    comment_match = re.search(r"COMMENT:(.*)", response)

    order_id = order_id_match.group(1) if order_id_match else None
    phone_num = phone_num_match.group(1) if phone_num_match else None
    comment = comment_match.group(1) if comment_match else None
    print("ORDER_ID:", order_id)
    print("PHONE_NUM:", phone_num)
    print("comment:", comment)

    if order_id is None:
        return "enter your order id:"
    elif phone_num is None:
        return "enter your phone number:"
    elif comment is None or comment.lower() == 'none':
        return "enter your comment:"
    else:
        result = comment_order(order_id, phone_num, comment)
        if result:
            cl.user_session.set("current_state", None)
            cl.user_session.set("chat_history", "")
            return "Your comment Successfuly added!"

        else:
            cl.user_session.set("chat_history", "")
            return "There is NO Order with this phone number, try again entering your id and phone number"


EXTRACTION_PROMPT = (
    "You are an assistant that extracts food ordering details from a user's query. "
    "Given the user's message, extract the following information in JSON format with exactly three keys: "
    "'food_name', 'restaurant_name', and 'query_scope'. The value for 'food_name' should be the food item mentioned. "
    "The value for 'restaurant_name' should be the name of the restaurant if the user explicitly mentions one; "
    "if not mentioned, return an empty string. The 'query_scope' value should be determined as follows:\n\n"
    "- If the user is asking about multiple restaurants (for example, using words like 'restaurants') or does not mention any restaurant, return \"all\".\n"
    "- If the user explicitly names a restaurant (for example, \"Milad and Sons Restaurant\"), return \"specific\".\n"
    "- If it is ambiguous whether the user wants a specific restaurant or all, return \"unsure\".\n\n"
    "For example:\n"
    "User: \"Which restaurants have Ghormeh Sabzi now and how much is it?\"\n"
    "Return: {\"food_name\": \"Ghormeh Sabzi\", \"restaurant_name\": \"\", \"query_scope\": \"all\"}\n\n"
    "User: \"How much is the pepperoni pizza at Milad and Sons Restaurant?\"\n"
    "Return: {\"food_name\": \"pepperoni pizza\", \"restaurant_name\": \"Milad and Sons Restaurant\", \"query_scope\": \"specific\"}\n\n"
    "Return only valid JSON. with nothing extra even saying jason"
)


def extract_query_details(user_input: str):
    """Extract food ordering details from the user's message using the LLM."""

    extraction_response = call_api(user_input, EXTRACTION_PROMPT).replace(
        'json', '').replace('```', '')
    match = re.search(r'({.*})', extraction_response, re.DOTALL)
    if match:
        json_text = match.group(1)
        try:
            extracted = json.loads(json_text)
            extracted["food_name"] = extracted.get("food_name", "").strip()
            extracted["restaurant_name"] = extracted.get(
                "restaurant_name", "").strip()
            extracted["query_scope"] = extracted.get(
                "query_scope", "").strip().lower()
        except json.JSONDecodeError as e:
            print("JSON decoding failed:", e)
            extracted = {"food_name": "",
                         "restaurant_name": "", "query_scope": "unsure"}
    else:
        print("No JSON object found in the response.")
        extracted = {"food_name": "",
                     "restaurant_name": "", "query_scope": "unsure"}

    return extracted


FINAL_ANSWER_PROMPT = (
    "You are Chatfood, an expert food ordering assistant. Given the user's original query, "
    "the details extracted (food name, restaurant name, and query scope), and the following database query results, "
    "compose a well-crafted and friendly answer. "
    "Make sure your answer explains which restaurants offer the food (and the price information) in a natural tone.\n\n"
    "User query: {user_query}\n\n"
    "Extracted details: {extracted}\n\n"
    "Database results: {db_results}\n\n"
    "Please provide your answer."
)


def generate_final_answer(user_query: str, extracted: dict, db_results: list):
    formatted_results = []
    for row in db_results:
        food, restaurant, price = row
        formatted_results.append({
            "food_name": food,
            "restaurant_name": restaurant,
            "price": price
        })

    user_message = FINAL_ANSWER_PROMPT.format(
        user_query=user_query,
        extracted=json.dumps(extracted, ensure_ascii=False),
        db_results=json.dumps(formatted_results, ensure_ascii=False)
    )
    system_prompt = "You are an assistant that crafts natural, friendly responses."
    return call_api(user_message, system_prompt)


def process_user_input_for_search(user_input):
    extracted = extract_query_details(user_input)
    print(extracted)
    food_name = extracted.get("food_name")
    restaurant_name = extracted.get("restaurant_name")
    query_scope = extracted.get("query_scope")

    if not food_name:
        clarification = "I'm sorry, I couldn't determine which food you are interested in. Could you please specify?"
        return clarification

    if query_scope == "unsure":
        clarification = "Do you have a specific restaurant in mind, or should I check all available options?"
        return clarification

    if query_scope == "specific" and not restaurant_name:
        clarification = "Could you please specify the restaurant you have in mind?"
        return clarification

    if query_scope == "specific":
        results = search_food_orders(food_name, restaurant_name)
    else:
        results = search_food_orders(food_name)

    final_answer = generate_final_answer(user_input, extracted, results)
    cl.user_session.set("current_state", None)
    cl.user_session.set("chat_history", "")
    return final_answer


def Local_Result(input_text, table, tf_model):
    query_vector = tf_model.encode(input_text).tolist()

    results = table.search(query_vector).limit(
        2).to_pydantic(DocumentEmbedding)
    context = ""
    for result in results:
        context += f"{result.chunk_text}\n"

    messages = [
        ("system",
         f"Based on this content, answer the user's question: {context}"),
        ("human", input_text),
    ]

    response = llm.invoke(messages)
    return response.content


def Internet_Result(input_text):
    TAVILY_API_KEY = load_key('TAVILY_API_KEY')
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily_client.search(input_text)

    messages = [
        ("system",
         f"If the context related to Food \ncontext:{response} Else Respose with-> Sorry let's just talk about foods! "),
        ("human", input_text),
    ]

    response = llm.invoke(messages)

    return response.content


@cl.on_message
async def respond(message: cl.Message):

    message = message.content
    response = ''
    if message.lower().strip() == 'exit':
        cl.user_session.set("current_state", None)
        cl.user_session.set("chat_history", "")
    current_state = cl.user_session.get("current_state", None)
    chat_history = cl.user_session.get("chat_history", "")
    if current_state is None:
        intend = handle_intend(message)
        cl.user_session.set("current_state", intend)
    cl.user_session.set("chat_history", chat_history + '\n' + message)
    current_state = cl.user_session.get("current_state", None)
    if current_state == 'CHECK_ORDER':
        response = handel_check_order(cl.user_session.get("chat_history", ""))
    elif current_state == 'CANCELL_ORDER':
        response = handel_cancel_order(cl.user_session.get("chat_history", ""))
    elif current_state == 'CREATE_COMMENT':
        response = handle_comment(cl.user_session.get("chat_history", ""))
    elif current_state == 'SEARCH_FOOD':
        response = process_user_input_for_search(
            cl.user_session.get("chat_history", ""))
    elif current_state == 'GENERAL_QUESTION':
        response = answering_general_questions(
            cl.user_session.get("chat_history", ""))
    response = f"ChatFood: {response}"
    if not cl.user_session.get("current_state", None) is None:
        cl.user_session.set("chat_history", cl.user_session.get(
            "chat_history", "") + '\n' + response)
    await cl.Message(content=response).send()
