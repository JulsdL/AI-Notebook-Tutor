# Use the official Python image from the Docker Hub
FROM python:3.11

# Create a new user with a specific UID and set it as the default user
RUN useradd -m -u 1000 user

# Switch to the new user
USER user

# Set environment variables
ENV HOME=/home/user \
  PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the requirements file and install the dependencies
COPY --chown=user ./requirements.txt $HOME/app/requirements.txt
RUN pip install --upgrade pip && \
  pip install -r requirements.txt

# Copy project files to the working directory
COPY --chown=user . $HOME/app

# Set PYTHONPATH to include the project directory
ENV PYTHONPATH=$HOME/app

# Set the command to run the application
CMD ["chainlit", "run", "notebook_tutor/app.py", "--port", "7860"]
