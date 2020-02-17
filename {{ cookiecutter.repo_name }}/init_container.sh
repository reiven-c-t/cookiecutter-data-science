PROJECT_DIR=/path/to/project/top

mkdir -p  mkdir -p /Users/reiven/Documents/Python
cd /Users/reiven/Documents/Python
ln -s /data ${PROJECT_DIR}
cd /Users/reiven/Documents/Python/${PROJECT_DIR}

pip install -r requirements.txt

python init_container.py

echo "export PYTHONPATH=/Users/reiven/Documents/Python/${PROJECT_DIR}/src" >> ~/.bashrc