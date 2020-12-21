#!/usr/bin/env bash
DEFAULT_INDEX_LEVEL="document"
INDEX_LEVEL=${1:-DEFAULT_INDEX_LEVEL}

INDEX_NAME="index_tf_idf_${INDEX_LEVEL}_level"
SIMILARITY_NAME="tf_idf"

echo "INDEX_NAME: $INDEX_NAME"
echo "SIMILARITY_NAME: $SIMILARITY_NAME"

curl -X PUT "localhost:9200/$INDEX_NAME?pretty" -H 'Content-Type: application/json' -d"
{
  \"settings\": {
    \"number_of_shards\": 1,
    \"similarity\": {
      \"$SIMILARITY_NAME\": {
        \"type\": \"scripted\",
        \"script\": {
          \"source\": \"double tf = Math.sqrt(doc.freq); double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0; double norm = 1/Math.sqrt(doc.length); return query.boost * tf * idf * norm;\"
        }
      }
    }
  }
}"