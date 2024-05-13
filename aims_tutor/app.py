import os
from dotenv import load_dotenv
import aims_tutor.chainlit_frontend as cl_frontend

# Load environment variables
load_dotenv()

# Main entry point
if __name__ == "__main__":
    cl_frontend.start_chat()
    cl_frontend.handle_user_query()
