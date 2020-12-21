#!/usr/bin/env bash
DEFAULT_INDEX_LEVEL="document"
INDEX_LEVEL=${1:-$DEFAULT_INDEX_LEVEL}

INDEX_NAME="index_bm25_${INDEX_LEVEL}_level"

echo "INDEX_NAME: $INDEX_NAME"
echo "SIMILARITY_NAME: BM25"

curl -X PUT "localhost:9200/$INDEX_NAME?pretty" -H 'Content-Type: application/json' -d"
{
  \"settings\": {
    \"number_of_shards\": 1
  }
}"