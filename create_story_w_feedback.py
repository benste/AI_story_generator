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
    global chatnames
    
    my_path = os.path.dirname(os.path.realpath(__file__))
    available_models = os.listdir(os.path.join(my_path, 'models'))
    available_models_str = "\n".join([f"{i}: {model}" for i, model in enumerate(available_models)])

    model_index = None
    while not model_index or not model_index.isdigit() or int(model_index) not in range(len(available_models)):
        model_index = input(f"Which model do you want to use:\n{available_models_str}\n")
        
    # Ask context size
    context_size = None
    while not context_size or not context_size.isdigit():
        context_size = input(f"What context size do you want to use (default = 4096):\n")
        if context_size == "":
            context_size = "4096"

    model_str = available_models[int(model_index)]
    model_path = os.path.join("models", model_str)
    print(f"loading {model_path} with {context_size} context size")
    llm = Llama(model_path=model_path, verbose=False, n_ctx=int(context_size))

    tokenizer_path = guess_tokenizer_path(model_str)
    correct_path = input(f'Guessed tokenizer HF path to be "{tokenizer_path}". Is this correct? (Y/n)')
    correct_path = correct_path.lower() == "y" or correct_path == ""
    
    if not correct_path or not tokenizer_path:
        tokenizer_path = input("Please provide the correct tokenizer HF path:\n")

    print(f"loading tokenizer {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Correct role names
    if "message['role'] == 'assistant'" in tokenizer.chat_template:
        chatnames = {"user": "user", "model": "assistant"}
    else:
        chatnames = {"user": "user", "model": "model"}
        

    return llm, tokenizer


def load_story() -> tuple:
    """Find available stories and load the one selected by user"""
    my_path = os.path.dirname(os.path.realpath(__file__))
    available_stories = os.listdir(os.path.join(my_path, 'stories'))
    available_stories = [story for story in available_stories if story.endswith(".pkl")]
    available_stories_str = "\n".join([f"{i}: {story}" for i, story in enumerate(available_stories)])
    
    story_index = None
    while not story_index or not story_index.isdigit() or int(story_index) not in range(len(available_stories_str)):
        story_index = input(f"Which story do you want to load:\n{available_stories_str}\n")

    story_path = os.path.join("stories", available_stories[int(story_index)])
    print(f"loading {story_path}")
    story_data = pickle.load(open(story_path, "rb"))
    
    print("story so far")
    for paragraph in story_data['paragraphs']:
        print(paragraph)
    
    return story_data['story_outline'], story_data['story_title'], story_data['paragraphs'], story_data['bullet_points'], story_data['paragraph_prompts']    

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
    output = llm(prompt, max_tokens=-1, stop=["<eos>"], echo=False, stream=True)
    story_outline = print_streamed(output)
    
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
    paragraph_prompts = [chat]
    prompt = generate_prompt(chat)
    output = llm(prompt, max_tokens=-1, echo=False, stream=True)
    paragraphs = [print_streamed(output)]

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
    for l in range(len(previous_paragraphs), 0, -1):
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
    paragraph_prompts.append(chat_length_constrainted(instructions['paragraph'], paragraphs[-3:]))
    prompt = generate_prompt(paragraph_prompts[-1])
    output = llm(prompt, max_tokens=-1, echo=False, stream=True)
    paragraphs.append(print_streamed(output))

    # New bullet points
    prompt = generate_prompt(chat_length_constrainted(instructions['bullet_points'], paragraphs[-3:]))
    output = llm(prompt, max_tokens=-1, stop=["<eos>"], echo=False) 
    bullet_points.append(output['choices'][0]['text']) 
    
    return paragraphs, bullet_points, paragraph_prompts


def change_last_paragraph(feedback):
    # Paragraph changed with feedback
    paragraph_prompts[-1].append({
        "role": chatnames['model'], 
        "content": paragraphs[-1],
        })
    paragraph_prompts[-1].append({
        "role": chatnames['user'], 
        "content": feedback,
        })
    prompt = generate_prompt(paragraph_prompts[-1])
    output = llm(prompt, max_tokens=-1, echo=False, stream=True) #, stop=["<eos>"]
    paragraphs[-1] = print_streamed(output)
    
    # New bullet points
    prompt = generate_prompt(chat_length_constrainted(instructions['bullet_points'], paragraphs[-3:]))
    output = llm(prompt, max_tokens=-1, stop=["<eos>"], echo=False) 
    bullet_points[-1] = output['choices'][0]['text']
    
    return paragraphs, bullet_points, paragraph_prompts


def create_story(story_outline=None, story_title=None, paragraphs=None, bullet_points=None, paragraph_prompts=None):
    """Main function to handle story generation."""

    continue_or_change = 'p'

    if not story_outline:
        # Create a story outline and title, ask the user if they are satisfied or new should be created
        good_outline = False
        while not good_outline:
            print("generating a story outline")
            story_title, story_outline = create_story_outline()
            good_outline = input(f'Title: {story_title}\nShould I create a different outline? (y/N)')
            good_outline = good_outline.lower() == "n" or good_outline == ""

    # Now that we have a title and outline, let's create a bunch of paragraphs
    story_ended = False
    while not story_ended:
        print("generating paragraph")
        if not paragraphs:
            # First paragraph and bullet points require a different prompt structure
            paragraphs, bullet_points, paragraph_prompts = prime_story()
        elif continue_or_change == 'c':
            # User wants to change the last paragraph
            paragraphs, bullet_points, paragraph_prompts = change_last_paragraph(feedback)
        else:
            paragraphs, bullet_points, paragraph_prompts = make_new_paragraph()
        
        # Save to text file
        with open(os.path.join("stories", f"{story_title.rstrip()}.txt"), 'w') as f:
            [f.write(paragraph + "\n") for paragraph in paragraphs]
            
        # Here we can ask the AI to change the last paragraph according to input from the user
        continue_or_change = None
        while not continue_or_change or continue_or_change not in ['c', 'p', 's', '']:
            continue_or_change = input(f"\nMake more paragraphs (p), Change this paragraph (c), or stop generating the story (s)- (default = p)\n").lower()
            if continue_or_change == "":
                continue_or_change = "p"
            
        if continue_or_change == 's':
            story_ended = True
            
        if continue_or_change == 'c':
            feedback = input(f"Tell the AI what to do different with the last paragraph\n")
        
            
        paragraphs[-1] = paragraphs[-1].rstrip("END").rstrip()

    return {
        "paragraph_prompts": paragraph_prompts,
        "story_title": story_title, 
        "story_outline": story_outline, 
        "bullet_points": bullet_points, 
        "paragraphs": paragraphs
    }
        
def main():
    global llm, tokenizer, instructions, story_outline, story_title, paragraphs, bullet_points, paragraph_prompts
    
    llm, tokenizer = load_model_and_tokenizer()
    
    # Load general instructions for creating a story in the style of Isaac Asimov
    instructions = json.load(open("instructions.json", "r"))
    
    # Ask to create a new story or start where left off
    new_story = None
    while not new_story or new_story not in ['c', 'n', '']:
        new_story = input(f"Continue previous story (c), or create a new story (n) (default = n)\n").lower()
    
    if new_story == 'c':
        story_outline, story_title, paragraphs, bullet_points, paragraph_prompts = load_story()
    
    # Start creating the story
    story_dict = None
    while not story_dict:
        story_dict = create_story(story_outline, story_title, paragraphs, bullet_points, paragraph_prompts)
        
    # Save the story
    with open(os.path.join("stories", f"{story_title.rstrip()}.pkl"), 'wb') as f:
        pickle.dump(story_dict, f)
        
if __name__ == "__main__":
    main()
