import requests

# Define the API endpoint
url = "http://localhost:5000/answer_query"

# Path to the PDF file
pdf_path = "QuillAudits- Code Cubicle Partnership MoU.pdf"

# Query
query = "Tell the experience of Manav"

# Create a dictionary with the PDF path and query
data = {"pdf_path": pdf_path, "query": query}

# Make a POST request to the API endpoint
response = requests.post(url, json=data)

# Print the response
print(response.json())
