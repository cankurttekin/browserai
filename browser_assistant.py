import asyncio
import os
import sys
import threading
import time
import logging
from typing import Optional, List
import concurrent.futures
from pathlib import Path

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr, BaseModel

from browser_use import Agent, BrowserConfig, Controller, ActionResult
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig

from config import GEMINI_API_KEY

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('browser_assistant.log')
    ]
)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

class BrowserResult(BaseModel):
    """Model for structured browser action results"""
    title: str
    url: str
    content: Optional[str] = None
    success: bool = True
    message: Optional[str] = None

class BrowserAssistant:
    def __init__(self):
        """Initialize the browser assistant with default settings"""
        self.browser = None
        self.browser_context = None
        self.controller = None
        self.agent = None
        self.agent_thread = None
        self.running = False
        self.mode = "interactive"  # Default to interactive mode
        self.loop = None
        self.browser_lock = threading.Lock()
        self.keep_browser_open = False  # New flag to control browser closure
        self.context_directory = None  # Directory for context files
        
        # Initialize the controller with custom actions
        self._initialize_controller()
        
    def _initialize_controller(self):
        """Initialize the controller with custom actions"""
        self.controller = Controller()
        
        # Register custom actions if needed
        @self.controller.registry.action('Take a screenshot and save it')
        async def take_screenshot(filename: str, browser):
            page = await browser.get_current_page()
            if not page:
                return ActionResult(error="No page is currently open")
            
            # Ensure the filename has a .png extension
            if not filename.endswith('.png'):
                filename = f"{filename}.png"
                
            await page.screenshot(path=filename)
            return ActionResult(
                extracted_content=f"Screenshot saved as {filename}",
                include_in_memory=True
            )
    
    def _initialize_browser(self):
        """Initialize the browser with appropriate configuration"""
        logger.info("Initializing new browser instance...")
        # Create a new browser instance with the configuration from the examples
        browser = Browser(
            config=BrowserConfig(
                headless=False,
                new_context_config=BrowserContextConfig(
                    viewport_expansion=0,
                )
            )
        )
        
        # Create a browser context
        browser_context = BrowserContext(
            config=BrowserContextConfig(viewport_expansion=0),
            browser=browser
        )
        
        return browser, browser_context
    
    def set_context_directory(self, directory: str):
        """Set the directory to scan for context files"""
        if os.path.exists(directory):
            self.context_directory = directory
            logger.info(f"Context directory set to: {directory}")
            return True
        else:
            logger.error(f"Directory does not exist: {directory}")
            return False
    
    def get_music_context(self) -> str:
        """Scan the context directory for music files and create a context string"""
        if not self.context_directory:
            return ""
            
        context = []
        try:
            # Walk through the directory
            for root, _, files in os.walk(self.context_directory):
                for file in files:
                    # Check for music file extensions
                    if file.lower().endswith(('.mp3', '.m4a', '.flac', '.wav')):
                        # Get the file name without extension
                        song_name = os.path.splitext(file)[0]
                        # Clean up the song name (remove underscores, etc.)
                        song_name = song_name.replace('_', ' ').replace('-', ' - ')
                        context.append(song_name)
            
            if context:
                context_str = "Based on these songs in my collection:\n"
                for song in context:
                    context_str += f"- {song}\n"
                context_str += "\nPlease find similar songs on YouTube and play one of them."
                return context_str
            
        except Exception as e:
            logger.error(f"Error reading context directory: {str(e)}")
        
        return ""
    
    def generate_browser_action(self, user_input: str, max_retries=3) -> str:
        """Generate a browser action from user input using Gemini with retry logic"""
        logger.info("Generating browser action from user input...")
        
        # Add music context if available and relevant
        if any(keyword in user_input.lower() for keyword in ["similar", "like", "recommendation"]):
            context = self.get_music_context()
            if context:
                user_input = f"{context}\n{user_input}"
                logger.info(f"Added music context to prompt: {context}")
        
        system_instruction = """You are a helpful AI assistant that generates browser actions based on user requests.
        Your task is to convert user requests into simple browser actions that can be executed.
        
        Respond with a clear, detailed instruction for a browser automation task.
        Make sure your response is specific and actionable.
        
        Important: Always include specific URLs when navigating to websites. For example, use 'Go to https://www.example.com' 
        instead of just 'Go to example.com'.
        
        For search tasks, specify the search engine with full URL, like 'Go to https://www.google.com and search for...'
        
        Keep your instructions simple and focused on one main task.
        
        For tasks involving media playback (like YouTube videos), make sure to indicate that the task should keep running
        after completion by adding 'and continue playing' at the end of the instruction.
        
        When recommending similar music based on a list of songs:
        1. Go directly to YouTube
        2. Search for one of the provided songs or a similar artist
        3. Look for related videos or recommendations
        4. Play one of the recommended videos
        5. Add 'and continue playing' at the end
        """
        
        for attempt in range(max_retries):
            try:
                # Use Gemini model with system instruction
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-pro",
                    system_instruction=system_instruction
                )
                
                # Log the complete prompt being sent to Gemini
                complete_prompt = f"System Instruction:\n{system_instruction}\n\nUser Input:\n{user_input}"
                logger.info(f"Sending prompt to Gemini:\n{complete_prompt}")
                
                # Generate content with the user prompt
                logger.info(f"Attempt {attempt + 1}/{max_retries} to generate content...")
                response = model.generate_content(user_input)
                
                # Log the AI's response
                logger.info(f"Gemini response:\n{response.text}")
                
                # Process the response to ensure it has proper navigation instructions
                action_text = response.text.strip()
                
                # If the action doesn't include a URL with https://, add a default search action
                if "https://" not in action_text and "http://" not in action_text:
                    if "similar" in user_input.lower() or "recommendation" in user_input.lower():
                        # For music recommendations, always go to YouTube
                        action_text = f"Go to https://www.youtube.com and {action_text}"
                    elif "search" in user_input.lower() or "find" in user_input.lower():
                        search_term = user_input.replace("search", "").replace("for", "").replace("find", "").strip()
                        action_text = f"Go to https://www.google.com and search for {search_term}"
                    else:
                        # Add a default navigation if needed
                        if action_text.lower().startswith("go to ") or action_text.lower().startswith("navigate to "):
                            # Extract the domain and add https://
                            parts = action_text.split(" ")
                            domain = parts[-1]
                            if not domain.startswith("http"):
                                domain = "https://" + domain
                            action_text = f"Go to {domain}"
                
                # For YouTube or media playback tasks, ensure we indicate to keep the browser open
                if "youtube" in action_text.lower() or "youtube" in user_input.lower():
                    if "and continue playing" not in action_text.lower():
                        action_text += " and continue playing"
                    self.keep_browser_open = True
                
                logger.info(f"Generated action: {action_text}")
                return action_text
                
            except Exception as e:
                logger.error(f"Error generating content (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retrying
                else:
                    raise Exception(f"Failed to generate browser action after {max_retries} attempts")
    
    async def execute_browser_action(self, action: str):
        """Execute a browser action using the browser-use Agent"""
        # Create a new browser instance for each action
        with self.browser_lock:
            browser, browser_context = self._initialize_browser()
            self.browser = browser
            self.browser_context = browser_context
        
        try:
            # Create the LLM with Gemini
            llm = ChatGoogleGenerativeAI(
                model='gemini-2.0-flash-exp', 
                api_key=SecretStr(GEMINI_API_KEY)
            )
            
            # Create the agent with the configured browser and controller
            logger.info("Creating agent with browser and controller...")
            agent = Agent(
                task=action,
                llm=llm,
                max_actions_per_step=4,
                browser_context=browser_context,  # Use browser_context instead of browser
                controller=self.controller,
                use_vision=True,  # Enable vision capabilities
            )
            
            with self.browser_lock:
                self.agent = agent
            
            # Run the agent
            self.running = True
            logger.info("Running agent...")
            history = await agent.run(max_steps=25)
            
            # Get the final result
            result = history.final_result()
            if result:
                logger.info(f"Task result: {result}")
                print(f"Result: {result}")
            
            print("Browser action completed")
            
            # For media playback or if keep_browser_open is True, wait for user input
            if self.keep_browser_open:
                logger.info("Keeping browser open for continued interaction...")
                print("\nBrowser will remain open. Type 'stop' to close it when done.")
                # Don't return here, let the browser stay open
            else:
                # For non-media tasks, wait briefly before closing
                await asyncio.sleep(2)
                await self.close_browser_internal()
            
        except Exception as e:
            logger.error(f"Error executing browser action: {str(e)}")
            print(f"Error executing browser action: {str(e)}")
        finally:
            self.running = False
    
    async def close_browser_internal(self):
        """Close the browser from within the same event loop"""
        with self.browser_lock:
            if self.browser_context:
                try:
                    logger.info("Closing browser context...")
                    await self.browser_context.close()
                    self.browser_context = None
                except Exception as e:
                    logger.error(f"Error closing browser context: {str(e)}")
            
            if self.browser:
                try:
                    logger.info("Closing browser...")
                    await self.browser.close()
                    self.browser = None
                except Exception as e:
                    logger.error(f"Error closing browser: {str(e)}")
    
    def close_browser(self):
        """Close the browser from the main thread"""
        with self.browser_lock:
            if self.browser or self.browser_context:
                logger.info("Initiating browser closure...")
                # Use a thread pool to run the coroutine
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_close_browser)
                    future.result()  # Wait for completion
    
    def _run_close_browser(self):
        """Run close_browser in a new event loop"""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.close_browser_internal())
        finally:
            loop.close()
    
    def start_action(self, action: str):
        """Start executing a browser action in a separate thread"""
        if self.agent_thread and self.agent_thread.is_alive():
            logger.warning("An action is already running")
            print("An action is already running. Please wait or stop it first.")
            return
        
        # Reset any existing browser/agent
        with self.browser_lock:
            if self.browser or self.browser_context:
                self.close_browser()
                self.browser = None
                self.browser_context = None
                self.agent = None
            
        # Create and start a new thread for the browser action
        logger.info("Starting new browser action thread...")
        self.agent_thread = threading.Thread(
            target=self._run_in_thread,
            args=(action,)
        )
        self.agent_thread.daemon = True  # Make thread daemon so it exits when main thread exits
        self.agent_thread.start()
    
    def _run_in_thread(self, action):
        """Run the browser action in a separate thread with its own event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.execute_browser_action(action))
        finally:
            loop.close()
    
    def stop_action(self):
        """Stop the currently running browser action"""
        with self.browser_lock:
            if self.agent and self.running:
                logger.info("Stopping browser action...")
                self.agent.stop()
                self.running = False
                self.keep_browser_open = False  # Reset the flag
                print("Browser action stopped")
            else:
                print("No browser action is currently running")
    
    def pause_action(self):
        """Pause the currently running browser action"""
        with self.browser_lock:
            if self.agent and self.running:
                logger.info("Pausing browser action...")
                self.agent.pause()
                print("Browser action paused")
            else:
                print("No browser action is currently running")
    
    def resume_action(self):
        """Resume the paused browser action"""
        with self.browser_lock:
            if self.agent:
                logger.info("Resuming browser action...")
                self.agent.resume()
                print("Browser action resumed")
            else:
                print("No browser action is available to resume")
    
    def close(self):
        """Close the browser and clean up resources"""
        with self.browser_lock:
            if self.running and self.agent:
                logger.info("Stopping running action before closure...")
                self.agent.stop()
                self.running = False
            
            if self.browser or self.browser_context:
                logger.info("Closing browser and cleaning up resources...")
                self.close_browser()
                self.browser = None
                self.browser_context = None
                self.agent = None
                self.keep_browser_open = False  # Reset the flag

def print_banner():
    """Print the application banner"""
    banner = """
    ╔══════════════════════════════════════════════╗
    ║                                              ║
    ║           AI BROWSER ASSISTANT               ║
    ║                                              ║
    ╚══════════════════════════════════════════════╝
    """
    print(banner)
    print("Type 'help' for available commands or 'exit' to quit.\n")

def print_help():
    """Print the help message with available commands"""
    help_text = """
    Available commands:
    - run [instruction]: Run a browser task with your instruction
    - stop: Stop the current browser task
    - pause: Pause the current browser task
    - resume: Resume a paused browser task
    - mode [autonomous/interactive]: Switch between autonomous and interactive modes
    - context [directory]: Set the directory to scan for music files
    - status: Check the current status of the browser assistant
    - help: Show this help message
    - exit: Quit the application
    
    Example instructions:
    - run go to https://www.google.com and search for browser-use
    - run navigate to https://github.com and find python projects
    - run find similar songs to what I listen to
    - run play the most viewed Tool song on YouTube
    """
    print(help_text)

def main():
    """Main function to run the browser assistant"""
    print_banner()
    
    # Create the browser assistant
    assistant = BrowserAssistant()
    
    # Default to interactive mode
    mode = "interactive"
    print(f"Current mode: {mode.capitalize()}")
    
    try:
        while True:
            user_input = input("\n> ")
            
            if user_input.lower() == "exit":
                print("Closing browser and exiting...")
                assistant.close()
                break
                
            elif user_input.lower() == "help":
                print_help()
                continue
                
            elif user_input.lower() == "status":
                status = "Running" if assistant.running else "Idle"
                print(f"Status: {status}")
                print(f"Mode: {mode.capitalize()}")
                if assistant.context_directory:
                    print(f"Context directory: {assistant.context_directory}")
                continue
                
            elif user_input.lower().startswith("context "):
                directory = user_input[8:].strip()
                if assistant.set_context_directory(directory):
                    print(f"Context directory set to: {directory}")
                else:
                    print("Invalid directory path")
                continue
                
            elif user_input.lower() == "stop":
                assistant.stop_action()
                continue
                
            elif user_input.lower() == "pause":
                assistant.pause_action()
                continue
                
            elif user_input.lower() == "resume":
                assistant.resume_action()
                continue
                
            elif user_input.lower().startswith("mode "):
                new_mode = user_input.lower().split("mode ")[1].strip()
                if new_mode in ["autonomous", "interactive"]:
                    mode = new_mode
                    print(f"Switched to {mode.capitalize()} mode")
                else:
                    print("Invalid mode. Use 'autonomous' or 'interactive'.")
                continue
            
            elif user_input.lower().startswith("run "):
                instruction = user_input[4:].strip()
                try:
                    print("Generating browser action...")
                    action = assistant.generate_browser_action(instruction)
                    print(f"\nAI Suggested Action: {action}")
                    
                    if mode == "interactive":
                        confirm = input("Execute? (y/n): ")
                        if confirm.lower() != "y":
                            continue
                    
                    print("Executing action...")
                    assistant.start_action(action)
                    
                except Exception as e:
                    print(f"Error: {str(e)}")
            else:
                print("Unknown command. Type 'help' for available commands.")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user. Closing browser and exiting...")
        assistant.close()
        sys.exit(0)

if __name__ == "__main__":
    main()