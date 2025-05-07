#!/usr/bin/env python3
"""
Script to generate a test summary from pytest HTML reports.
This is used in the CI/CD pipeline to create a summary of test results.
"""

import os
import re
import json
from bs4 import BeautifulSoup
from datetime import datetime

def parse_html_report(file_path):
    """Parse the pytest HTML report and extract test results."""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Extract summary information
    summary = {}
    summary_table = soup.find('p', text=re.compile('Run duration'))
    if summary_table and summary_table.find_next('table'):
        for row in summary_table.find_next('table').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) == 2:
                key = cells[0].text.strip()
                value = cells[1].text.strip()
                summary[key] = value
    
    # Extract test results
    tests = []
    results_table = soup.find('h2', text=re.compile('Results'))
    if results_table and results_table.find_next('table'):
        for row in results_table.find_next('table').find_all('tr')[1:]:  # Skip header row
            cells = row.find_all('td')
            if len(cells) >= 3:
                test_name = cells[2].text.strip()
                test_result = "passed" if "passed" in row.get('class', []) else "failed"
                tests.append({
                    "name": test_name,
                    "result": test_result
                })
    
    return {
        "summary": summary,
        "tests": tests
    }

def generate_summary():
    """Generate a summary of test results from both guest and pro modules."""
    guest_results = parse_html_report("../guest/test-results.html")
    pro_results = parse_html_report("../pro/test-results.html")
    
    # Create summary dictionary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "guest": guest_results,
        "pro": pro_results
    }
    
    # Save as JSON
    with open("../test-summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Generate Markdown report
    md = "# Bhrigu.ai Test Summary\n\n"
    md += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Create a table of all test cases
    md += "## Test Cases\n\n"
    md += "| Module | Test Name | Result |\n"
    md += "|--------|-----------|--------|\n"
    
    # Add guest test cases
    if guest_results:
        for test in guest_results["tests"]:
            emoji = "✅" if test["result"] == "passed" else "❌"
            md += f"| Guest | {test['name']} | {emoji} {test['result']} |\n"
    
    # Add pro test cases
    if pro_results:
        for test in pro_results["tests"]:
            emoji = "✅" if test["result"] == "passed" else "❌"
            md += f"| Pro | {test['name']} | {emoji} {test['result']} |\n"
    
    # Add module summaries
    if guest_results:
        md += "\n## Guest Module Summary\n\n"
        md += "| Metric | Value |\n"
        md += "|--------|-------|\n"
        for key, value in guest_results["summary"].items():
            md += f"| {key} | {value} |\n"
    else:
        md += "\n## Guest Module\n\nNo test results available.\n\n"
    
    if pro_results:
        md += "\n## Pro Module Summary\n\n"
        md += "| Metric | Value |\n"
        md += "|--------|-------|\n"
        for key, value in pro_results["summary"].items():
            md += f"| {key} | {value} |\n"
    else:
        md += "\n## Pro Module\n\nNo test results available.\n\n"
    
    # Save as Markdown
    with open("../test-summary.md", "w", encoding="utf-8") as f:
        f.write(md)
    
    print("Test summary generated successfully.")

if __name__ == "__main__":
    generate_summary()