import json
import os
import pickle
from typing import Any, Iterator, Union

from llama_cpp import CreateCompletionResponse, Llama
from transformers import AutoTokenizer


class StoryGenerator:
    def __init__(self) -> None:
        self.story_outline = ""
        self.story_title = ""
        self.paragraphs: list = []
        self.bullet_points: list = []
        self.paragraph_prompts: list = []
        self.initialized = False
        self.use_bullet_points = True

        # INITIALIZE #
        # Load instructions for creating a story
        self.inquire_and_load_story_type()

        # Load model and tokenizer
        self.load_model_and_tokenizer()

        # Correct role names
        if "message['role'] == 'assistant'" in self.tokenizer.chat_template:
            self.chatnames = {"user": "user", "model": "assistant"}
        else:
            self.chatnames = {"user": "user", "model": "model"}

        # Ask if bullet points should be created after each paragraph
        self.inquire_bullet_points()

    def inquire_and_load_story_type(self) -> None:
        my_path = os.path.dirname(os.path.realpath(__file__))
        available_instructions = os.listdir(os.path.join(my_path, "instructions"))
        available_instructions_str = "\n".join(
            [f"{i}: {instructions}" for i, instructions in enumerate(available_instructions)]
        )

        instruction_index = None
        while (
            not instruction_index
            or not instruction_index.isdigit()
            or int(instruction_index) not in range(len(available_instructions))
        ):
            instruction_index = input(
                f"Which model do you want to use:\n{available_instructions_str}\n"
            )

        model_str = available_instructions[int(instruction_index)]
        self.instructions: dict = json.load(
            open(os.path.join(my_path, "instructions", model_str), "r")
        )

    def guess_tokenizer_path(self, model_str: str) -> Union[str, None]:
        """Guesses the Hugging Face tokenizer path based on the model string."""
        model_str = model_str.lower()
        if model_str.startswith("phi-3"):
            size_str, context_str = "", ""
            for size in ["mini", "small", "medium"]:
                if size in model_str:
                    size_str = size
                    break

            for context in ["4k", "128k"]:
                if context in model_str:
                    context_str = context
                    break

            return f"microsoft/Phi-3-{size_str}-{context_str}-instruct"

        if model_str.startswith("gemma-2"):
            size_str = ""
            for size in ["9b", "27b"]:
                if size in model_str:
                    size_str = size
                    break

            return f"google/gemma-2-{size_str}-it"

        return None

    def inquire_bullet_points(self) -> None:
        use_bullet_points = None
        while use_bullet_points is None or use_bullet_points not in ["y", "n", ""]:
            use_bullet_points = input(
                """\n\nI can create bullet-points of the important events after each paragraph
to assist in creating the next paragraph.
This can help tracking important events many paragraphs back, but is no guarantee and
it also adds more compute time.
With large context length it is recommended to not use bullet-points.
Should I update story bullet-points after each paragraph? (Y/n)"""
            ).lower()
        self.use_bullet_points = use_bullet_points == "y" or ""

        if not self.use_bullet_points:
            self.instructions["paragraph"] = self.instructions["paragraph_no_bulletpoints"]
            self.bullet_points = [None]

    def load_model_and_tokenizer(self) -> None:
        """Loads the selected model and tokenizer."""
        my_path = os.path.dirname(os.path.realpath(__file__))
        available_models = [
            model
            for model in os.listdir(os.path.join(my_path, "models"))
            if not model.startswith(".")
        ]
        available_models_str = "\n".join(
            [f"{i}: {model}" for i, model in enumerate(available_models)]
        )

        model_index = None
        while (
            not model_index
            or not model_index.isdigit()
            or int(model_index) not in range(len(available_models))
        ):
            model_index = input(f"Which model do you want to use:\n{available_models_str}\n")

        # Ask context size
        context_size = None
        while not context_size or not context_size.isdigit():
            context_size = input(
                "\n\nWhat context size do you want to use (default = 4096, max = 32000):\n"
            )
            if context_size == "":
                context_size = "4096"

        self.context_size: int = min([int(context_size), 32000])

        model_str = available_models[int(model_index)]
        model_path = os.path.join(my_path, "models", model_str)
        print(f"loading {model_path} with {context_size} context size")
        self.llm: Llama = Llama(model_path=model_path, verbose=False, n_ctx=self.context_size)

        tokenizer_path = self.guess_tokenizer_path(model_str)
        correct_path = input(
            f'\n\nGuessed tokenizer HF path to be "{tokenizer_path}". Is this correct? (Y/n)'
        )
        correct_path = correct_path.lower() == "y" or correct_path == ""

        if not correct_path or not tokenizer_path:
            tokenizer_path = input("Please provide the correct tokenizer HF path:\n")

        print(f"loading tokenizer {tokenizer_path}")
        self.tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_path)

    def save_story(self) -> None:
        # Save to text file
        with open(os.path.join("stories", f"{self.story_title.rstrip()}.txt"), "w") as f:
            [
                f.write(paragraph.rstrip("END").rstrip().lstrip() + "\n")
                for paragraph in self.paragraphs
            ]

        # Save to pickle
        with open(os.path.join("stories", f"{self.story_title.rstrip()}.pkl"), "wb") as f:
            pickle.dump(
                {
                    "paragraph_prompts": self.paragraph_prompts,
                    "story_title": self.story_title,
                    "story_outline": self.story_outline,
                    "bullet_points": self.bullet_points,
                    "paragraphs": self.paragraphs,
                },
                f,
            )

    def load_story(self) -> None:
        """Find available stories and load the one selected by user"""
        my_path = os.path.dirname(os.path.realpath(__file__))
        available_stories = os.listdir(os.path.join(my_path, "stories"))
        available_stories = [story for story in available_stories if story.endswith(".pkl")]
        available_stories_str = "\n".join(
            [f"{i}: {story}" for i, story in enumerate(available_stories)]
        )

        story_index = None
        while (
            not story_index
            or not story_index.isdigit()
            or int(story_index) not in range(len(available_stories_str))
        ):
            story_index = input(f"Which story do you want to load:\n{available_stories_str}\n")

        story_path = os.path.join(my_path, "stories", available_stories[int(story_index)])
        print(f"loading {story_path}")
        story_data = pickle.load(open(story_path, "rb"))

        print("story so far")
        for paragraph in story_data["paragraphs"]:
            print(paragraph)

        self.story_outline = story_data["story_outline"]
        self.story_title = story_data["story_title"]
        self.paragraphs = story_data["paragraphs"]
        self.bullet_points = story_data["bullet_points"]
        self.paragraph_prompts = story_data["paragraph_prompts"]
        self.initialized = True

    def generate_prompt(self, chat: list) -> str:
        """Generates a prompt using the tokenizer chat template."""
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def print_streamed(self, stream: Iterator[CreateCompletionResponse]) -> str:
        full_text = ""
        for s in stream:
            token = s["choices"][0]["text"]
            print(token, end="", flush=True)
            full_text += token

        return full_text

    def create_story_outline(self) -> None:
        """Creates a story outline and title."""
        # outline
        chat = [
            {
                "role": self.chatnames["user"],
                "content": self.instructions["initialization"]["outline"],
            }
        ]
        prompt = self.generate_prompt(chat)
        output = self.llm(prompt, max_tokens=-1, stop=["<eos>"], echo=False, stream=True)
        self.story_outline = self.print_streamed(output)  # type: ignore

        # title
        chat.append({"role": self.chatnames["model"], "content": self.story_outline})
        chat.append(
            {
                "role": self.chatnames["user"],
                "content": self.instructions["initialization"]["title"],
            }
        )
        prompt = self.generate_prompt(chat)
        self.story_title = self.llm(prompt, max_tokens=-1, stop=["<eos>"], echo=False)["choices"][  # type: ignore  # noqa: E501
            0
        ][
            "text"
        ]

    def prime_story(self) -> None:
        """Primes the story with an initial paragraph and bullet points."""
        # First paragraph
        chat = [
            {
                "role": self.chatnames["user"],
                "content": self.instructions["initialization"]["outline"],
            },
            {"role": self.chatnames["model"], "content": self.story_outline},
            {
                "role": self.chatnames["user"],
                "content": self.instructions["initialization"]["title"],
            },
            {"role": self.chatnames["model"], "content": self.story_title},
            {
                "role": self.chatnames["user"],
                "content": self.instructions["initialization"]["paragraph"],
            },
        ]
        self.paragraph_prompts = [chat]
        prompt = self.generate_prompt(chat)
        output = self.llm(prompt, max_tokens=-1, echo=False, stream=True)
        self.paragraphs = [self.print_streamed(output)]  # type: ignore

        # First bullet points
        if self.use_bullet_points:
            chat.append({"role": self.chatnames["model"], "content": self.paragraphs[0]})
            chat.append(
                {
                    "role": self.chatnames["user"],
                    "content": self.instructions["initialization"]["bullet_points"],
                }
            )
            prompt = self.generate_prompt(chat)
            output = self.llm(prompt, max_tokens=-1, stop=["<eos>"], echo=False)
            self.bullet_points = [output["choices"][0]["text"]]  # type: ignore

        self.initialized = True

    def chat_length_constrainted(self, prompt: str) -> list:
        """Ensures the prompt does not exceed the context length."""
        # Guess number of previous paragraphs in prompt that wouldn't exceeding context
        max_num_paragraphs = int(round((self.context_size - 1000) / 500))

        # Now make sure it doesn't actually exceed context length
        for le in range(min([max_num_paragraphs, len(self.paragraphs)]), 0, -1):
            chat = [
                {
                    "role": self.chatnames["user"],
                    "content": prompt.format(
                        f_outline=self.story_outline,
                        f_bullet_points=self.bullet_points[-1],
                        f_num_paragraphs=le,
                        f_latest_paragraph="\n".join(self.paragraphs[-le:]),
                    ),
                }
            ]
            tokenized = self.tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True
            )
            if len(tokenized) < self.context_size:
                return chat

        return [
            {
                "role": self.chatnames["user"],
                "content": prompt.format(
                    f_outline=self.story_outline,
                    f_bullet_points=self.bullet_points[-1],
                    f_num_paragraphs=0,
                    f_latest_paragraph="",
                ),
            }
        ]

    def make_new_paragraph(self) -> None:
        """Creates a new paragraph and corresponding bullet points."""
        # New paragraph
        self.paragraph_prompts.append(self.chat_length_constrainted(self.instructions["paragraph"]))
        prompt = self.generate_prompt(self.paragraph_prompts[-1])
        output = self.llm(prompt, max_tokens=-1, echo=False, stream=True)
        self.paragraphs.append(self.print_streamed(output))  # type: ignore

        # New bullet points
        if self.use_bullet_points:
            prompt = self.generate_prompt(
                self.chat_length_constrainted(self.instructions["bullet_points"])
            )
            output = self.llm(prompt, max_tokens=-1, stop=["<eos>"], echo=False)
            self.bullet_points.append(output["choices"][0]["text"])  # type: ignore

        self.save_story()

    def change_last_paragraph(self, feedback: str) -> None:
        # Paragraph changed with feedback
        self.paragraph_prompts[-1].append(
            {
                "role": self.chatnames["model"],
                "content": self.paragraphs[-1],
            }
        )
        self.paragraph_prompts[-1].append(
            {
                "role": self.chatnames["user"],
                "content": feedback,
            }
        )
        prompt = self.generate_prompt(self.paragraph_prompts[-1])
        output = self.llm(prompt, max_tokens=-1, echo=False, stream=True)
        self.paragraphs[-1] = self.print_streamed(output)  # type: ignore

        # New bullet points
        if self.use_bullet_points:
            prompt = self.generate_prompt(
                self.chat_length_constrainted(self.instructions["bullet_points"])
            )
            output = self.llm(prompt, max_tokens=-1, stop=["<eos>"], echo=False)
            self.bullet_points[-1] = output["choices"][0]["text"]  # type: ignore

        self.save_story()
