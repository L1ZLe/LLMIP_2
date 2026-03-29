#!/bin/bash

# ERCOT Manual Snapshot System
# A manual version control system for your ERCOT project code, scripts, and configurations

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_DIR="$HOME/LLMIP"
SAVE_DIR="$PROJECT_DIR/save"
COMMIT_SCRIPT="$SAVE_DIR/llmip-snapshot-ai.py"
LOCK_FILE="$SAVE_DIR/.snapshot.lock"
ENV_FILE="$PROJECT_DIR/.env"

# Load environment variables from .env
if [ -f "$ENV_FILE" ]; then
    # Read .env file and export variables (handles "key = value" format)
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        
        # Parse "key = value" or "key=value" format (also remove carriage returns)
        key=$(echo "$line" | cut -d'=' -f1 | tr -d ' \r')
        value=$(echo "$line" | cut -d'=' -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | tr -d '\r')
        
        # Export the variable (only if key is not empty)
        if [ -n "$key" ]; then
            export "$key=$value"
        fi
    done < "$ENV_FILE"
else
    echo -e "${RED}ERROR: .env file not found at $ENV_FILE${NC}"
    exit 1
fi

# Check for GitHub_key
if [ -z "$GitHub_key" ]; then
    echo -e "${RED}ERROR: GitHub_key not found in .env file${NC}"
    exit 1
fi

# Repo URL with token
REPO_URL="https://${GitHub_key}@github.com/L1ZLe/LLMIP_2.git"

# Ensure we're in the project directory
cd "$PROJECT_DIR" || {
    echo -e "${RED}ERROR: Cannot cd to $PROJECT_DIR${NC}"
    exit 1
}

# Check if git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Not a git repository${NC}"
    echo -e "${YELLOW}Run: cd $PROJECT_DIR && git init${NC}"
    exit 1
fi

# Main function
main() {
    case "${1:-status}" in
        save_update|update|save)
            shift
            save_commit "$@"
            ;;
        status)
            show_status
            ;;
        history)
            show_history "${2:-10}"
            ;;
        revert_update|revert)
            shift
            revert_commit "$@"
            ;;
        *)
            show_help
            ;;
    esac
}

# Save a commit with auto-generated or custom message
save_commit() {
    # Check for lock file (prevent concurrent snapshots)
    if [ -f "$LOCK_FILE" ]; then
        echo -e "${YELLOW}WARNING: Another snapshot in progress. Lock file found at $LOCK_FILE${NC}"
        exit 1
    fi

    # Create lock
    touch "$LOCK_FILE"

    # Trap to remove lock on exit
    trap 'rm -f "$LOCK_FILE"' EXIT

    local commit_msg
    if [ $# -gt 0 ]; then
        # Custom message provided
        commit_msg="$*"
    else
        # Auto-generate commit message
        commit_msg=$(python3 "$COMMIT_SCRIPT" auto)
    fi

    echo -e "${BLUE}=== Saving ERCOT Snapshot ===${NC}"
    echo -e "${GREEN}Message: $commit_msg${NC}"
    echo ""

    # Stage all changes
    git add -A

    # Show what's being committed
    echo -e "${YELLOW}Changes to be committed:${NC}"
    git diff --cached --stat

    echo ""
    # Commit
    git commit -m "$commit_msg"

    # Update remote URL with token (in case it changed)
    git remote set-url origin "$REPO_URL"

    # Push to GitHub
    echo -e "${BLUE}Pushing to GitHub...${NC}"
    if git push origin main 2>&1; then
        echo -e "${GREEN}✓ Push successful${NC}"
    else
        echo -e "${RED}✗ Push failed${NC}"
        echo -e "${YELLOW}Try: git push origin main${NC}"
    fi

    echo -e "${GREEN}✓ Snapshot saved${NC}"
}

# Show current git status
show_status() {
    echo -e "${BLUE}=== ERCOT Status ===${NC}"
    echo ""
    git status --short

    echo ""
    echo -e "${YELLOW}Recent commits (last 5):${NC}"
    git log --oneline -5
}

# Show commit history
show_history() {
    local count="${1:-10}"
    echo -e "${BLUE}=== ERCOT Snapshot History ($count commits) ===${NC}"
    echo ""
    git log --oneline --stat --pretty=format:"%h | %s | %ar" | head -n "$count"
}

# Revert to a previous commit
revert_commit() {
    if [ $# -eq 0 ]; then
        echo -e "${RED}ERROR: Please provide commit hash${NC}"
        echo "Usage: $0 revert_update <commit_hash>"
        show_history 20
        exit 1
    fi

    local commit_hash="$1"

    echo -e "${YELLOW}=== Reverting ERCOT Snapshot ===${NC}"
    echo -e "${RED}This will revert to commit: $commit_hash${NC}"
    echo ""

    # Show commit details
    git show --stat "$commit_hash"

    echo ""
    read -p "$(echo -e ${YELLOW}Confirm revert? [y/N]: ${NC})" -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Create revert commit
        git revert --no-edit "$commit_hash"

        echo -e "${YELLOW}Revert created. Review changes and push:${NC}"
        echo "1. git status"
        echo "2. git diff"
        echo "3. $0 save_update 'Reverted to $commit_hash'"
    else
        echo -e "${BLUE}Revert cancelled${NC}"
    fi
}

# Show help message
show_help() {
    cat << EOF
${GREEN}ERCOT Snapshot System${NC}
Manual version control for ERCOT project with AI commit messages.

${BLUE}Usage:${NC}
    $0 <command> [arguments]

${BLUE}Commands:${NC}
    ${GREEN}save_update, update, save${NC}    Save current state with auto or custom message
        $0 save_update "Added new congestion model"
        $0 save_update "Updated feature engineering pipeline"

    ${GREEN}status${NC}                          Show current git status

    ${GREEN}history${NC}                         Show commit history
        $0 history          Show last 10 commits
        $0 history 20      Show last 20 commits

    ${GREEN}revert_update, revert${NC}       Revert to previous commit
        $0 revert_update <commit_hash>

    ${GREEN}help${NC}                             Show this help message

${BLUE}Examples:${NC}
    # Before major changes
    $0 save_update "Before refactoring data pipeline"

    # After training model
    $0 save_update "Added Random Forest for spread prediction"

    # After updating notebook
    $0 save_update "Updated EDA with congestion heatmap"

    # Check status
    $0 status

    # View history
    $0 history

    # Revert if something breaks
    $0 history
    $0 revert_update abc1234

${BLUE}Project:${NC}
    Directory: $PROJECT_DIR
    Repository: $REPO_URL
    GitHub: https://github.com/L1ZLe/LLMIP_2

${BLUE}What Gets Tracked:${NC}
    ✓ Code (ercot_main/, models/)
    ✓ Notebooks (notebooks/)
    ✓ Data (data/raw/, data/processed/, data/final/)
    ✓ Reports (reports/)
    ✓ Documentation (docs/)
    ✓ Scripts (scripts/)
    ✗ scripts/save/ directory (snapshot scripts ignored)

EOF
}

# Run main function
main "$@"
