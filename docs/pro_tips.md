# Pro Tips for AI Codebase Exploration

If you are a newbie developer wanting to use an AI assistant to learn a new codebase, the best approach is to start from the **top down**—moving from high-level architecture to specific implementation details. 

Here is a step-by-step prompting guide you can use (in this project or any future project) to quickly get up to speed:

### Step 1: Grasp the "Big Picture" (High-Level Architecture)
Before reading any code, you want to know what the project does and how the folders are laid out.
* **Prompt 1:** `"Summarize what this project does in 2-3 sentences based on the README. Who is the intended user?"`
* **Prompt 2:** `"Can you generate a markdown tree of the root directory and explain what each folder is responsible for?"`
* **Prompt 3:** `"What are the core technologies, frameworks, and libraries used in this repository?"`

### Step 2: Find the Entry Points (Execution Flow)
Codebases are much easier to read when you know where the program "starts".
* **Prompt 4:** `"Where are the main entry points for this application (e.g., the scripts a user would run to train or test the model)?"`
* **Prompt 5:** `"Walk me through the execution flow of tools/train.py. What happens step-by-step when I run it?"`
* **Prompt 6:** `"How is configuration handled in this project? Where do the default parameters live?"`

### Step 3: Understand the Core Logic & Models
Once you know where the script starts, dive into the actual "brains" of the application.
* **Prompt 7:** `"Explain the architecture of the neural network model in lib/models/hrnet.py in simple terms. How does it handle high-resolution inputs?"`
* **Prompt 8:** `"What does the forward() method do in the main model class?"`
* **Prompt 9:** `"Can you explain the mathematical goal or loss function used to train this model?"`

### Step 4: Trace the Data Flow (Inputs & Outputs)
Machine learning and data-heavy projects are all about the data pipeline.
* **Prompt 10:** `"How is the dataset loaded in lib/datasets/ceph.py? What format are the annotations in?"`
* **Prompt 11:** `"Walk me through the data augmentation pipeline. What transformations are applied to the images before they hit the model?"`
* **Prompt 12:** `"What is the exact shape/format of the input tensor going into the model, and what does the output prediction look like?"`

### Step 5: Hands-on Debugging & Experimentation
The best way to learn is by making small changes or running the code in debug mode.
* **Prompt 13:** `"I want to add a simple print() statement to see the shape of the data entering the model. Where is the best place to put this?"`
* **Prompt 14:** `"How can I run a tiny, dummy test on this code just to ensure my environment is set up correctly without training for hours?"`
* **Prompt 15:** `"Can you write a small Python snippet using this codebase's inference functions to test a single image?"`

### 💡 Pro-Tip for Interacting with AI:
Whenever the AI explains a file, you can always ask it to go deeper: 
> *"You mentioned `function.py` handles the training loop. Can you explain lines 50-80 of that file to me line-by-line?"*
