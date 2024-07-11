from story_generator import StoryGenerator


def create_story(my_story_generator: StoryGenerator):
    """Main story generation loop"""

    if not my_story_generator.initialized:
        # Create a story outline and title, ask the user if they are satisfied or new should be created
        good_outline = False
        while not good_outline:
            print("generating a story outline")
            my_story_generator.create_story_outline()
            good_outline = input(f'Title: {my_story_generator.story_title}\nShould I create a different outline? (y/N)')
            good_outline = good_outline.lower() == "n" or good_outline == ""

    # Now that we have a title and outline, let's create a bunch of paragraphs
    story_ended = False
    continue_or_change = 'p'
    while not story_ended:
        print("generating paragraph")
        if not my_story_generator.initialized:
            # First paragraph and bullet points require a different prompt structure
            my_story_generator.prime_story()
        elif continue_or_change == 'c':
            # User wants to change the last paragraph
            my_story_generator.change_last_paragraph(feedback)
        else:
            my_story_generator.make_new_paragraph()
            
        # Here we can ask if the AI should change the last paragraph according to input from the user
        continue_or_change = None
        while continue_or_change is None or continue_or_change not in ['c', 'p', 's', '']:
            continue_or_change = input(f"\nMake more paragraphs (p), Change this paragraph (c), or stop generating the story (s)- (default = p)\n").lower()
            
        if continue_or_change == 's':
            story_ended = True
            
        if continue_or_change == 'c':
            feedback = input(f"Tell the AI what to do different with the last paragraph\n")
            
        
def main():
    my_story_generator = StoryGenerator()
    
    # Ask to create a new story or start where left off
    new_story = None
    while new_story is None or new_story not in ['c', 'n', '']:
        new_story = input(f"Continue previous story (c), or create a new story (n) (default = n)\n").lower()
        if new_story == 'c':
            my_story_generator.load_story()
    
    # Start creating the story
    create_story(my_story_generator)
        
if __name__ == "__main__":
    main()
