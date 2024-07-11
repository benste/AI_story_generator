from story_generator import StoryGenerator


def create_story(my_story_generator: StoryGenerator) -> bool:
    """Main story generation loop"""

    if not my_story_generator.initialized:
        # Create a story outline and title, ask the user if it's fine or a new should be created
        good_outline = False
        while not good_outline:
            print("generating a story outline")
            my_story_generator.create_story_outline()
            how_is_outline = input(
                f"Title: {my_story_generator.story_title}\nShould I create a different outline? (y/N)"  # noqa: E501
            )
            good_outline = how_is_outline.lower() == "n" or how_is_outline == ""

    # Now that we have a title and outline, let's create a bunch of paragraphs
    story_ended = False
    while not story_ended:
        # How many should be generated this round
        num_new_paragraphs = None
        while not num_new_paragraphs or not num_new_paragraphs.isdigit():
            num_new_paragraphs = input(
                "How many new paragraphs should I create? (you'll be able to make more later)\n"
            )

        num_new_paragraphs = int(num_new_paragraphs)
        goal_num_paragraphs = len(my_story_generator.paragraphs) + num_new_paragraphs

        for _ in range(num_new_paragraphs):
            print(
                f"\n**generating paragraph {len(my_story_generator.paragraphs)+1}/{goal_num_paragraphs}\n"  # noqa: E501
            )

            if not my_story_generator.initialized:
                # First paragraph and bullet points require a different prompt structure
                my_story_generator.prime_story()
            else:
                my_story_generator.make_new_paragraph()

            if my_story_generator.paragraphs[-1].endswith("END"):
                print(
                    "I STOPPED GENERATING MORE PARAGRAPHS BECAUSE I MIGHT HAVE FOUND THE END OF THE STORY"  # noqa: E501
                )
                break

        story_ended = ""
        while story_ended.lower() not in ["y", "n", "yes", "no", "r"]:
            story_ended = input("\nShould I create more paragraphs? (Y/n/r, r=restart)")
            if story_ended == "":
                story_ended = "y"

        if story_ended.lower() == "r":
            return False
        else:
            story_ended = story_ended[0].lower() == "n"

    return True


def main() -> None:
    story_made = False
    while not story_made:
        my_story_generator = StoryGenerator()

        # Ask to create a new story or start where left off
        new_story = None
        while new_story is None or new_story not in ["c", "n", ""]:
            new_story = input(
                "\n\nContinue previous story (c), or create a new story (n) (default = n)\n"
            ).lower()
            if new_story == "c":
                my_story_generator.load_story()

        # Start creating the story
        story_made = create_story(my_story_generator)


if __name__ == "__main__":
    main()
