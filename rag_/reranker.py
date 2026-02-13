def rerank(query, docs):
    scored = [(doc, doc.lower().count(query.lower())) for doc in docs]
    scored.sort(key=lambda x:x[1], reverse=True)
    return [s[0] for s in scored]
