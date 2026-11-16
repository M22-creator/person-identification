"""Live web view of the person identification pipeline.

    python -m person_id.web --source 4      # then open http://localhost:8000

server.py       camera loop + HTTP server (standard library only)
auto_enroll.py  saves new faces to the gallery as person_N
index.html      the page itself
"""
