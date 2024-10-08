#!/bin/bash

# Prompt for email
echo "Enter your Git email:"
read git_email

# Prompt for name
echo "Enter your Git name:"
read git_name

# Set Git config
git config --global user.email "$git_email"
git config --global user.name "$git_name"

echo "Git config has been updated:"
echo "Email: $git_email"
echo "Name: $git_name"

# Verify the changes
echo -e "\nVerifying changes:"
git config --global --get user.email
git config --global --get user.name