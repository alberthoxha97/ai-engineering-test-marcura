# AI Engineer Challenge: Charter Party Document Parser

## Overview

Your task is to build a Python application that parses a maritime charter party document (PDF) and extracts legal clauses in a structured format using LLM capabilities.

## Document

**Source:** https://shippingforum.wordpress.com/wp-content/uploads/2012/09/voyage-charter-example.pdf

This is a voyage charter party agreement – a standard maritime contract used in shipping. The document contains:
- **Part I**: Particulars/Details (skip this section)
- **Part II**: Legal clauses with numbered provisions (**Pages 6–39**)

## Requirements

1. **Extract legal clauses from Part II** of the document (**Pages 6–39**)
2. **For each clause, extract:**
   - `id`: The clause identifier (e.g., "1", "2", "3", etc.)
   - `title`: The clause title/heading
   - `text`: The full clause text content

3. **Output the extracted clauses** in a structured JSON format. 
4. Do not include strike-thru text.
5. Clauses should be returned in the order they appear in the document.

**DO include the output json in the final submission**

### Technical Requirements

1. You can use any LLM of your choice.
2. Focus on code quality. It should show your python skills!
3. We should be able to run your code locally.

**DO NOT include your API key in the submission!**

PS. Please publish the solution to your GitHub and invite us to review it.

Any questions shoot and good luck!