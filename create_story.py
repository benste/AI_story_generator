import json
import os
import pickle

from llama_cpp import Llama
from transformers import AutoTokenizer


def guess_tokenizer_path(model_str: str) -> str:
    """Guesses the Hugging Face tokenizer path based on the model string."""
    model_str = model_str.lower()
    if model_str.startswith("phi-3"):
        size_str, context_str = "", ""
        for size in ['mini', 'small', 'medium']:
            if size in model_str:
                size_str = size
                break
        
        for context in ['4k', '128k']:
            if context in model_str:
                context_str = context
                break
            
        return f"microsoft/Phi-3-{size_str}-{context_str}-instruct"
        
    if model_str.startswith("gemma-2"):
        size_str = ""
        for size in ['9b', '27b']:
            if size in model_str:
                size_str = size
                break
        
        return f"google/gemma-2-{size_str}-it"
    
    return None


def load_model_and_tokenizer():
    """Loads the selected model and tokenizer."""
    my_path = os.path.dirname(os.path.realpath(__file__))
    available_models = os.listdir(os.path.join(my_path, 'models'))
    available_models_str = "\n".join([f"{i}: {model}" for i, model in enumerate(available_models)])

    model_index = None
    while not model_index or not model_index.isdigit() or int(model_index) not in range(len(available_models)):
        model_index = input(f"Which model do you want to use:\n{available_models_str}\n")

    model_str = available_models[int(model_index)]
    model_path = os.path.join("models", model_str)
    print(f"loading {model_path}")
    llm = Llama(model_path=model_path, verbose=False, n_ctx=4096)

    tokenizer_path = guess_tokenizer_path(model_str)
    correct_path = input(f'Guessed tokenizer HF path to be "{tokenizer_path}". Is this correct? (Y/n)')
    correct_path = correct_path.lower() == "y" or correct_path == ""
    
    if not correct_path or not tokenizer_path:
        tokenizer_path = input("Please provide the correct tokenizer HF path:\n")

    print(f"loading tokenizer {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    return llm, tokenizer, model_str


def generate_prompt(chat):
    """Generates a prompt using the tokenizer chat template."""
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def print_streamed(stream) -> str:
    full_text = ""
    for s in stream:
        token = s['choices'][0]['text']
        print(token, end="", flush=True)
        full_text += token
        
    return full_text


def create_story_outline() -> tuple:
    """Creates a story outline and title."""
    # outline
    chat = [{"role": chatnames['user'], "content": instructions['initialization']['outline']}]
    prompt = generate_prompt(chat)
    stream = llm(prompt, max_tokens=-1, stop=["<eos>"], echo=False, stream=True)
    story_outline = print_streamed(stream)

    # title
    chat.append({"role": chatnames['model'], "content": story_outline})
    chat.append({"role": chatnames['user'], "content": instructions['initialization']['title']})
    prompt = generate_prompt(chat)
    story_title = llm(prompt, max_tokens=-1, stop=["<eos>"], echo=False)['choices'][0]['text']
    
    return story_title, story_outline


def prime_story() -> tuple:
    """Primes the story with an initial paragraph and bullet points."""
    # First paragraph
    chat = [
        {"role": chatnames['user'], "content": instructions['initialization']['outline']},
        {"role": chatnames['model'], "content": story_outline},
        {"role": chatnames['user'], "content": instructions['initialization']['title']},
        {"role": chatnames['model'], "content": story_title},
        {"role": chatnames['user'], "content": instructions['initialization']['paragraph']},
    ]
    paragraph_prompts = [generate_prompt(chat)]
    stream = llm(paragraph_prompts[-1], max_tokens=-1, echo=False, stream=True)
    paragraphs = [print_streamed(stream)]

    # First bullet points
    chat.append({"role": chatnames['model'], "content": paragraphs[0]})
    chat.append({"role": chatnames['user'], "content": instructions['initialization']['bullet_points']})
    prompt = generate_prompt(chat)
    output = llm(prompt, max_tokens=-1, stop=["<eos>"], echo=False) 
    bullet_points = [output['choices'][0]['text']]

    return paragraphs, bullet_points, paragraph_prompts


def chat_length_constrainted(prompt, previous_paragraphs, max_tokens=4096) -> str:
    """Ensures the prompt does not exceed the context length."""
    # Most number of previous paragraphs in prompt without exceeding context. Max 6 paragraphs
    for l in range(min([6, len(previous_paragraphs)]), 0, -1):
        chat = [{
            "role": chatnames['user'], 
            "content": prompt.format(
                f_outline=story_outline,
                f_bullet_points=bullet_points[-1],
                f_num_paragraphs=l,
                f_latest_paragraph="\n".join(previous_paragraphs[:l])
            )
        }]
        tokenized = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
        if len(tokenized) < max_tokens:
            return chat
        
    return [{
        "role": chatnames['user'], 
        "content": prompt.format(
            f_outline=story_outline,
            f_bullet_points=bullet_points[-1],
            f_num_paragraphs=l,
            f_latest_paragraph=""
            )
        }]
        

def make_new_paragraph():
    """Creates a new paragraph and corresponding bullet points."""
    # New paragraph
    paragraph_prompts.append(generate_prompt(chat_length_constrainted(instructions['paragraph'], paragraphs[-3:])))
    stream = llm(paragraph_prompts[-1], max_tokens=-1, stop=["<eos>"], echo=False, stream=True)
    paragraphs.append(print_streamed(stream))

    # New bullet points
    bullet_prompt = generate_prompt(chat_length_constrainted(instructions['bullet_points'], paragraphs[-3:]))
    output = llm(bullet_prompt, max_tokens=-1, stop=["<eos>"], echo=False) 
    bullet_points.append(output['choices'][0]['text']) 
    
    return paragraphs, bullet_points, paragraph_prompts


def create_story():
    """Main function to handle story generation."""
    global story_outline, story_title, paragraphs, bullet_points, paragraph_prompts
    
    # Create a story outline and title, ask the user if they are satisfied or new should be created
    good_outline = False
    while not good_outline:
        print("generating a story outline")
        story_title, story_outline = create_story_outline()
        good_outline = input(f'I created the following story outline:\nTitle: {story_title}\n{story_outline}\nShould I create a different outline? (y/N)')
        good_outline = good_outline.lower() == "n" or good_outline == ""

    # Now that we have a title and outline, let's create a bunch of paragraphs
    paragraphs = []
    bullet_points = []
    story_ended = False
    while not story_ended:
        # How many should be generated this round
        num_new_paragraphs = None
        while not num_new_paragraphs or not num_new_paragraphs.isdigit():
            num_new_paragraphs = input("How many new paragraphs should I create? (you'll be able to make more later)\n")

        num_new_paragraphs = int(num_new_paragraphs)
        goal_num_paragraphs = len(paragraphs) + num_new_paragraphs

        for _ in range(num_new_paragraphs):
            if not paragraphs:
                # First paragraph and bullet points require a different prompt structure
                paragraphs, bullet_points, paragraph_prompts = prime_story()
            else:
                paragraphs, bullet_points, paragraph_prompts = make_new_paragraph()

            print(f"generated paragraph {len(paragraphs)}/{goal_num_paragraphs}")

            if paragraphs[-1].endswith("END"):
                print("I STOPPED GENERATING MORE PARAGRAPHS BECAUSE I MIGHT HAVE FOUND THE END OF THE STORY")
                break

        story_ended = ""
        while story_ended.lower() not in ['y', 'n', 'yes', 'no', 'r']:
            story_ended = input("\nShould I create more paragraphs? (Y/n/r, r=restart)")
            if story_ended == "": story_ended = "y"
            
        if story_ended.lower() == "r":
            return None
        else:
            story_ended = story_ended[0].lower() == "n"
            
        paragraphs[-1] = paragraphs[-1].rstrip("END")

    return {
        "paragraph_prompts": paragraph_prompts,
        "story_title": story_title, 
        "story_outline": story_outline, 
        "bullet_points": bullet_points, 
        "paragraphs": paragraphs
    }
        
def main():
    global llm, tokenizer, chatnames, instructions
    
    llm, tokenizer, model_str = load_model_and_tokenizer()
    
    # Chat names differ for gemma and phi models
    if model_str.lower().startswith("phi-3"):
        chatnames = {"user": "user", "model": "assistant"}
    else:
        chatnames = {"user": "user", "model": "model"}
    
    # Load general instructions for creating a story in the style of Isaac Asimov
    instructions = json.load(open("instructions.json", "r"))
    
    # Ask to create a new story or start where left off
    
    # Start creating the story
    story_dict = None
    while not story_dict:
        story_dict = create_story()
        
    # Save the story
    with open(os.path.join("stories", f"{story_title.rstrip()}.pkl"), 'wb') as f:
        pickle.dump(story_dict, f)
        
    # Save as text

if __name__ == "__main__":
    main()
