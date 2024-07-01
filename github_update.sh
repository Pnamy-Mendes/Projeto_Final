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

# Function to configure Git settings for large files
configure_git_settings() {
    git config --global http.postBuffer 524288000  # 500MB
    git config --global http.maxRequestBuffer 100M
}

# Function to get the current branch name
get_current_branch() {
    git branch --show-current 2>/dev/null || echo "Master"
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
    git push --force -u origin "$(get_current_branch)"
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

# Function to clear Git cache for a specific file
clear_git_cache() {
    local cache_file="$1"
    if git ls-files --error-unmatch "$cache_file" > /dev/null 2>&1; then
        echo "Removing $cache_file from Git index..."
        git rm --cached "$cache_file"
        echo "$cache_file has been removed from the Git index."
    else
        echo "$cache_file is not in the Git index."
    fi
}

# Ensure cache file is in .gitignore
echo "Ensuring cache file is in .gitignore..."
echo "cache/UTK_age_gender_race/data_cache.npz" >> .gitignore
echo ".gitignore updated."

# Clear cache file from Git index
clear_git_cache "cache/UTK_age_gender_race/data_cache.npz"

# Check if the repository is already initialized
if [ -d .git ]; then
    echo "Git repository already initialized. Updating repository..."
    configure_git_identity
    configure_git_settings
    update_repo
else
    echo "Git repository not found. Initializing repository..."
    configure_git_identity
    configure_git_settings
    initialize_repo
fi

echo "Done."
