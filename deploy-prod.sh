#!/bin/bash

python3 -m venv venv

source venv/bin/activate
python3 -m pip install -q gdown

gdown --continue --folder "https://drive.google.com/drive/folders/1GgUoDd7fpo3XWT7HQ_5q5RxJYa5YFvPU" -O $PWD

EXCLUDE_FILE="exclude-sync.txt"

if [ $# -ne 4 ]; then
    echo "Usage: $0 <remote_user> <remote_host> <ssh_port> <ssh-key>"
    echo "Example: $0 username 192.168.1.100 22 ~/.ssh/id-rsa"
    exit 1
fi

REMOTE_USER=$1
REMOTE_HOST=$2
SSH_PORT=$3
SSH_KEY=$4

SOURCE="./"
REMOTE_PATH="/home/"$REMOTE_USER"/projects/pneumonia"
ABSOLUTE_REMOTE_PATH="/home/"$REMOTE_USER"/projects/pneumonia"

SERVICE_NAME="pneumonia"
START_COMMAND="$ABSOLUTE_REMOTE_PATH/venv/bin/streamlit run app.py --server.port 8504"

ssh -i $SSH_KEY -p $SSH_PORT "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_PATH" && \
rsync -avzP --delete -e "ssh -o \"StrictHostKeyChecking no\" -i $SSH_KEY -p $SSH_PORT" --exclude-from="$EXCLUDE_FILE" "$SOURCE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

if [ $? -eq 0 ]; then
    echo "Transfer completed successfully"
else
    echo "Transfer failed"
fi

ssh -i $SSH_KEY -p $SSH_PORT "$REMOTE_USER@$REMOTE_HOST" bash -s << EOF
# Setup virtual environment and install dependencies
cd $REMOTE_PATH
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate

# Create systemd service file
sudo bash -c "cat > /etc/systemd/system/${SERVICE_NAME}.service" << 'ENDSERVICE'
[Unit]
Description=Pneumonia Detector
After=network.target

[Service]
Type=simple
User=${REMOTE_USER}
WorkingDirectory=${ABSOLUTE_REMOTE_PATH}
Environment="PATH=${ABSOLUTE_REMOTE_PATH}/venv/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=${START_COMMAND}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
ENDSERVICE

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}.service
sudo systemctl start ${SERVICE_NAME}.service
sudo systemctl status ${SERVICE_NAME}.service
EOF

echo "Deployment and service setup completed"
exit 1
