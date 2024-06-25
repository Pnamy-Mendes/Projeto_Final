#!/bin/bash

# Define variables
REPO_URL="https://github.com/Pnamy-Mendes/Projeto_Final.git"

# Function to check and set Git user identity
configure_git_identity() {
    if ! git config --global user.name >/dev/null; then
        echo "Please enter your Git user name:"
        read -r user_name
        git config --global user.name "$user_name"
    fi

    if ! git config --global user.email >/dev/null; then
        echo "Please enter your Git user email:"
        read -r user_email
        git config --global user.email "$user_email"
    fi
}

# Function to get the current branch name
get_current_branch() {
    git branch --show-current 2>/dev/null || echo "main"
}

# Function to initialize the git repository
initialize_repo() {
    echo "Initializing Git repository..."
    git init
    echo "Adding all files to the repository..."
    git add .
    echo "Committing the files..."
    git commit -m "Initial commit"
    echo "Adding remote repository..."
    git remote add origin "$REPO_URL"
    echo "Pushing to remote repository..."
    git push -u origin "$(get_current_branch)"
}

# Function to update the repository
update_repo() {
    echo "Fetching latest changes from remote..."
    git fetch origin "$(get_current_branch)"
    echo "Pulling latest changes from remote..."
    git pull origin "$(get_current_branch)"
    echo "Adding all files to the repository..."
    git add .
    echo "Committing the changes..."
    git commit -m "Update project"
    echo "Pushing changes to remote repository..."
    git push origin "$(get_current_branch)"
}

# Check if the repository is already initialized
if [ -d .git ]; then
    echo "Git repository already initialized. Updating repository..."
    configure_git_identity
    update_repo
else
    echo "Git repository not found. Initializing repository..."
    configure_git_identity
    initialize_repo
fi

echo "Done."
