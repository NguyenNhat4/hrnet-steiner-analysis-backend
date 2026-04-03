# The Daily AI Code Mastery Routine

If you want to ensure you are compounding your knowledge every single day when learning codebases with an AI assistant, you need a structured approach. It's easy to just ask the AI to "fix the bug," but to actually *grow* as a developer, you need intention.

Here are daily practices to ensure you are always getting better:

## 1. The "Never Copy-Paste Blindly" Rule
This is the single most important habit. When the AI gives you a block of code:
* **Read every line.**
* **If you don't understand it, ask the AI to explain it.**
* *Actionable Task:* Before you paste the code, manually type it out (at least the first few times). Muscle memory and forced reading slow you down just enough to understand what you are doing.

## 2. The "Break It to Make It" Practice
A great way to learn is to intentionally break working code.
* Take a working script (e.g., `tools/test.py`).
* Change a configuration value to something ridiculous (e.g., batch size of `0` or `10000`).
* Run it. Look at the error.
* **Prompt the AI:** `"I just changed the batch size to 10000 and got this Out Of Memory error: [paste error]. Why does batch size affect memory and where in the code is the memory actually allocated?"`

## 3. Daily Code Tracing (15 Minutes)
Pick one small feature or data pipeline per day and trace it from beginning to end. 
* **Example:** "Today, I will figure out exactly how an image is loaded, augmented, and turned into a Tensor."
* Use your debugger, or use `print()` statements to check the type and shape of data at every step.
* **Prompt the AI:** `"I am tracing the data augmentation in lib/utils/transforms.py. Can you explain what the affine matrix logic is actually doing to the image array?"`

## 4. Reverse Engineer the AI's Solutions
When the AI solves a problem for you, don't just move on. Ask *how* it arrived at the solution.
* **Prompt the AI:** `"That fix worked perfectly. However, I want to learn. Can you explain your thought process? What cues in my error message led you to realize the data type was a float instead of a long?"`
* This teaches you the *heuristics* of debugging, not just the syntax of Python.

## 5. End-of-Day Rubber Duck Summary
At the end of your coding session, "teach" the AI what you learned. This forces you to consolidate your knowledge.
* **Prompt the AI:** `"Today I learned that PyTorch datasets subclass torch.utils.data.Dataset and require the __len__ and __getitem__ methods. In this project, that happens in ceph.py. Did I understand that correctly? Is there any nuance I missed?"`

## 6. Curate a Personal "TIL" (Today I Learned) Log
Keep a running Markdown file (e.g., `docs/TIL.md` or a Notion page).
* Every time you learn a new concept (e.g., "what is a bounding box regression loss"), write a 2-sentence summary.
* Every time you encounter a nasty bug and solve it, write down the error message and the fix. 
* *Why this works:* When you see the bug again in three weeks, you will know exactly how to fix it without asking the AI.

## The Mental Shift
Stop treating the AI as an *oracle* that exists just to give you the exact file modifications to make. Start treating the AI as a **Senior Staff Engineer pairing with you**.

A junior dev asks: *"What is the code to fix this?"*
A junior dev *who is leveling up* asks: *"Why did my architecture choice cause this bottleneck, and what pattern should I study so I don't do this again?"*
