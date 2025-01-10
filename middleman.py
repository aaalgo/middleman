#!/usr/bin/env python3
import openai
from typing import Literal, Optional
from pydantic import BaseModel
import subprocess
import json

SYSTEM_PROMPT = """
You are the middleman AI, which sits between the user and the bash command line of a recent Ubuntu system. 
Your input will be interleaved user input, the command you generate and system feedback from executing any of these commands.

Your behavior:
1. You always generate a response as a JSON object with the following schema:

{
  "type": "plain" | "command" | "terminate",
  "content": "<string displayed to the user>",
  "command": "<bash command or null>",
  "confirm": <true|false>,
  "terminate": <true|false>
}

2. The field 'type' can be:
   - "plain": A message that only displays 'content' to the user.
   - "command": A message containing a 'command' field to be executed on the system.
   - "terminate": A message indicating the conversation should end after displaying 'content' to the user.

3. If you provide a "command" of type "command":
   - The "content" field must always include an explanation or reason.
   - The "confirm" field specifies whether we need explicit user confirmation before running the command.
     - If "confirm" is true, the user will be asked: "Run this command? (yes/no)".
     - If the user declines, we will pass that feedback into your next input.
   - If "confirm" is false, we will run the command without asking for confirmation.

4. If "type" is "terminate":
   - We will display the "content" message to the user and then end the session.

5. You must always respond in valid JSON. 

Your role is to:
- Interpret user requests and any system feedback (stdout, stderr, or user-declined commands).
- Provide either an explanation, a new command, or end the conversation, based on the user’s needs.

Remember:
- Always respond in valid JSON.
- Do not include any additional text or markdown outside the JSON structure.
- The commands you generate should run without any user inputs.
"""

class Response(BaseModel):
    type: Literal["plain", "command", "terminate"]
    content: str
    command: Optional[str]
    confirm: bool

client = openai.OpenAI()

# ANSI color codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

def prompt_user():
    return input(">> ")

def format_message(user_input=None, command=None, stdout=None, stderr=None, declined=False):
    """
    Constructs a message with sections only if they have content:
      --- User Input:
      --- COMMAND:
      --- STDOUT:
      --- STDERR:
      --- DECLINED:
    """
    msg = ""
    if user_input:
        msg += f"--- User Input:\n{user_input}\n"
    if command:
        msg += f"--- COMMAND:\n{command}\n"
    if declined:
        msg += "--- DECLINED\n"
    if stdout:
        msg += f"--- STDOUT:\n{stdout}\n"
    if stderr:
        msg += f"--- STDERR:\n{stderr}\n"
    return msg

def ask_chatgpt(context):
    """
    Sends 'context' to OpenAI ChatCompletion and expects a JSON response:
      {
        "type": "plain" | "command" | "terminate",
        "content": "...",
        "command": "...",        # if type == "command"
        "confirm": bool,         # if type == "command"
        "terminate": bool        # optional if type == "terminate"
      }
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=context,
        response_format=Response
    )
    parsed = response.choices[0].message.parsed
    content = response.choices[0].message.content
    return parsed, content

def run_command(command):
    """
    Executes a shell command, returning (stdout, stderr, returncode).
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def main():
    context = []
    carry_over_message = None  # Stores message for the next loop iteration

    # Optional system instruction:
    context.append({"role": "system", "content": SYSTEM_PROMPT})

    while True:
        # Either use carry_over_message as user input or prompt the user
        if carry_over_message:
            user_input = carry_over_message
            carry_over_message = None
        else:
            user_input = prompt_user()

        # Add the user's message to the context
        context.append({"role": "user", "content": user_input})

        # Ask the AI for a response
        ai_response, content = ask_chatgpt(context)
        print(ai_response.content)

        # Add the AI's response back to the context
        context.append({"role": "assistant", "content": content})

        msg_type = ai_response.type
        if msg_type == "terminate":
            print("Session terminated by AI.")
            break

        elif msg_type == "command":
            command = ai_response.command
            confirm = ai_response.confirm

            # Ask user for confirmation if needed
            if confirm:
                user_confirmation = input(f"Run this command? {command} (yes/no): ")
                if user_confirmation.lower() != "yes":
                    declined_message = format_message(command=command, declined=True)
                    carry_over_message = declined_message
                    continue

            # Execute the command
            print(f"{YELLOW}{command}{RESET}")
            stdout, stderr, returncode = run_command(command)

            # Color-code the output
            colored_stdout = f"{GREEN}{stdout}{RESET}" if stdout else ""
            colored_stderr = f"{RED}{stderr}{RESET}" if stderr else ""

            # Format the output message
            output_message = format_message(
                command=command,
                stdout=stdout if stdout else None,
                stderr=stderr if stderr else None
            )
            # Display the color-coded output to the user
            if stdout:
                print(f"{YELLOW}--- STDOUT:{RESET}")
                print(colored_stdout, end="")
            if stderr:
                print(f"{YELLOW}--- STDERR:{RESET}")
                print(colored_stderr, end="")

            # Add the raw (uncolored) message to context for AI
            context.append({"role": "assistant", "content": output_message})

            # Use the output as carry-over message for the next iteration
            carry_over_message = output_message

        # If type == "plain", just loop again

if __name__ == "__main__":
    main()

