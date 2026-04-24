import asyncio
import boto3
import json
import os
import psycopg2
from dotenv import load_dotenv
from pathlib import Path



load_dotenv(dotenv_path=Path(__file__).parent.parent.parent/".env")


#--- Configs
EMBED_MODEL = 'amazon.titan-embed-text-v2:0'
TOP_K = 50 #Fetch top 50 from each search methods
FINAL_TOP_K = 5 #Return top 5 to users after mergin
RRF_K = 60 #RRF constants - standard value, dampens rank effect


#---- AWS CLIENTS
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)


#---- POSTGRESQL CONNECTION
def get_conn():
    return psycopg2.connect(
        host='127.0.0.1',
        port=5432,
        dbname='meridian',
        user='postgres',
        password='password',


    )

#-- EMBED QUERY
async def embed_query(text:str) ->list[float]:
    response = await asyncio.to_thread(
        bedrock.invoke_model,
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text[:8000],
                        "dimensions":1024,
                        "normalize":True})
    )
    result = json.loads(response['body'].read())
    return result['embedding']



#--- DENSE SEARCH
def dense_search(cur,embedding:list[float],space_key:str =None)->list[dict]:
    if space_key:
        sql = f"""
                    SELECT chunk_id, text,page_title, space_key, url,
                    1 - (embedding <=> %s::vector) as score
                    FROM chunks
                    WHERE space_key = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """
        cur.execute(sql,(str(embedding),space_key, str(embedding),TOP_K))
    else:
        sql = """
            SELECT chunk_id, text, page_title, space_key, url,
                   1 - (embedding <=> %s::vector) as score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        cur.execute(sql, (str(embedding), str(embedding), TOP_K))

    rows = cur.fetchall()
    
    # Convert to list of dicts for easier handling
    return [
        {
            "chunk_id":   row[0],
            "text":       row[1],
            "page_title": row[2],
            "space_key":  row[3],
            "url":        row[4],
            "score":      float(row[5]),
            "source":     "dense"
        }
        for row in rows
    ]



#SPARSE SEARCH
def sparse_search(cur, query:str, space_key:str = None) -> list[dict]:
    if space_key:
        sql = """
                    SELECT chunk_id, text,page_title, space_key,url,
                    ts_rank(fts,plainto_tsquery('english',%s)) as score
                    FROM chunks
                    WHERE fts @@ plainto_tsquery('english',%s)
                    AND space_key = %s
                    ORDER BY score DESC
                    LIMIT %s;

                    """
        cur.execute(sql, (query,query, space_key, TOP_K))

    else:
        sql = """
            select chunk_id, text, page_title, space_key, url,
            ts_rank(fts, plainto_tsquery('english',%s)) as score
            FROM chunks
            WHERE fts @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s;
        """
        cur.execute(sql, (query, query, TOP_K))

    rows = cur.fetchall()

    return [
        {
            "chunk_id":   row[0],
            "text":       row[1],
            "page_title": row[2],
            "space_key":  row[3],
            "url":        row[4],
            "score":      float(row[5]),
            "source":     "sparse"
        }
        for row in rows
    ]



#RRF MERGE 
def reciprocal_rank_fusion(
        dense_results:list[dict],
        sparse_results:list[dict]
) -> list[dict]:
    scores = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk['chunk_id']
        if cid not in scores:
            scores[cid] = {"chunk":chunk, "rrf_score": 0.0}

        scores[cid]["rrf_score"] += 1.0 / (rank + RRF_K)


    #process sparse results
    for rank, chunk in enumerate(sparse_results):
        cid = chunk['chunk_id']
        if cid not in scores:
            scores[cid] = {"chunk":chunk, "rrf_score":0.0}

        scores[cid]["rrf_score"] += 1.0 / (rank + RRF_K)

    sorted_chunks = sorted(
            scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
    
    result = []

    for item in sorted_chunks[:FINAL_TOP_K]:
        chunk = item["chunk"].copy()
        chunk['rrf_score'] = round(item['rrf_score'],6)
        result.append(chunk)

    return result




### MAIN Retrieval Function

async def retrieve(query:str, space_key:str = None)->list[dict]:
    embedding = await embed_query(query)

    conn = get_conn()
    cur = conn.cursor()

    dense_results = dense_search(cur, embedding, space_key)
    sparse_results = sparse_search(cur,query, space_key)

    cur.close()
    conn.close()


    #Step 3: Merge with RRF
    final_results = reciprocal_rank_fusion(dense_results,sparse_results)

    return final_results


#Test

async def test_retrieval():
    test_queries = [
        "How does kafka handle replication",
        "what is the flink checkpoint mechanism",
        "How to schedule tasks in apache airflow"
    ]
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = await retrieve(query)
        for i, chunk in enumerate(results):
            print(f"\n  Result {i+1}:")
            print(f"  Title    : {chunk['page_title']}")
            print(f"  Space    : {chunk['space_key']}")
            print(f"  RRF Score: {chunk['rrf_score']}")
            print(f"  URL      : {chunk['url']}")
            print(f"  Preview  : {chunk['text'][:100]}...")


if __name__ == "__main__":
    asyncio.run(test_retrieval())