def map_keywords(document, keywords):
    #the doucment is in the form of a docstring
    
    paragraphs = document.strip().split("\n\n")  # Split the document into paragraphs using double newline

    mapped_paragraphs = {keyword.lower(): [] for keyword in keywords}

    for paragraph in paragraphs:
        for keyword in mapped_paragraphs.keys():
            if keyword in paragraph.lower():
                mapped_paragraphs[keyword].append(paragraph)

    return mapped_paragraphs
