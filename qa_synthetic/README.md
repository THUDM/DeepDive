# Synthetic QA Data Generation with Knowledge Graphs

This folder contains scripts and workflows to generate **synthetic QA datasets** based on **random walk on Knowledge Graph**. The final merged outputs contribute to the official dataset:

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-DeepDive-blueviolet)](https://huggingface.co/datasets/zai-org/DeepDive)

## 📦 Step 1: Install Dependencies

### Install MongoDB Tools (Ubuntu)

```bash
# Add the official MongoDB source
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" \
    | tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# Install the toolkit and verify
apt-get update && apt-get install -y mongodb-database-tools
mongoimport --version

# Install Python dependencies
pip install -r requirements.txt
```

## 📚 Step 2: Download Knowledge Graph Source

Download [KILT](https://github.com/facebookresearch/KILT) dataset (Wikipedia snapshot, ~35GB):

```bash
wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
```

## 🗄️ Step 3: Start Database & Import Data

Run MongoDB container:

```bash
docker run -d --name mongo -p 27017:27017 -v mongodata:/data/db mongo:6
```

Import knowledge source (10–20 min):

```bash
mongoimport --uri "mongodb://127.0.0.1:27017" \
  --db kilt --collection knowledgesource \
  --file kilt_knowledgesource.json
```

Create index for faster queries:

```bash
docker exec -it mongo mongosh --eval \
  'db.getSiblingDB("kilt").knowledgesource.createIndex({ wikipedia_title: "text", text: "text" })'
```

Test connection:

```bash
python3 kilt_query.py
```

## 🔑 Step 4: Configure API Keys

First, create the .env file

```bash
touch .env
```

then add your credentials:

```bash
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENROUTER_API_KEY=sk-your-openrouter-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## 🎲 Step 5: Random Walk on Knowledge Graph

Sample multi-hop paths:

```bash
python3 random_walk_kilt.py
```

Outputs will be stored in `./random_walk_outputs/`.

## 💬 Step 6: Generate QA Data

Use Gemini-2.5-pro to create Q&A pairs:

```bash
python3 generate_qa.py \
  --input ./random_walk_outputs/random_walk_architecture.jsonl \
  --output ./random_walk_outputs/random_walk_architecture_qa.jsonl
```
