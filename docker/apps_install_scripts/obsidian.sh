#!/bin/bash
set -e

AGENT_HOME="/home/agent"

# Install needed packages
apt update
apt install -y fuse libfuse2

# Download the obsidian AppImage
wget -4 https://github.com/obsidianmd/obsidian-releases/releases/download/v1.7.7/Obsidian-1.7.7.AppImage -O ${AGENT_HOME}/Obsidian.AppImage

chmod +x ${AGENT_HOME}/Obsidian.AppImage

# Manually extract obsidian
cd ${AGENT_HOME}

# TODO: Check
${AGENT_HOME}/Obsidian.AppImage --appimage-extract

# Correct Ownership
chown -R agent:agent ${AGENT_HOME}/squashfs-root
chown agent:agent ${AGENT_HOME}/Obsidian.AppImage

# Create wrapper script with --no-sandbox flag
# Add symbolic link (So that it becomes visible for the agent)
cat <<'EOF' >/usr/local/bin/obsidian
#!/bin/bash
exec /home/agent/squashfs-root/obsidian --no-sandbox "$@"
EOF

chmod +x /usr/local/bin/obsidian

# Create an alias with no-sandbox for easy access
# echo 'alias obsidian="/home/agent/squashfs-root/obsidian --no-sandbox"' >>${AGENT_HOME}/.bashrc

# Setup pre-defined global obsidian config files so that it works with the pre-defined vaults
cp -r /workspace/PC-Canary/tests/context_data/obsidian/obsidian/ ${AGENT_HOME}/.config/

# Fix permissions
chown -R agent:agent ${AGENT_HOME}/.config

cd /workspace
