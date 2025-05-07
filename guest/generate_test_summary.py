import xml.etree.ElementTree as ET

def generate_summary(xml_file, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    test_cases = []
    
    for testcase in root.findall(".//testcase"):
        name = testcase.get("name")
        classname = testcase.get("classname")
        time = float(testcase.get("time", 0))
        status = "Pass" if testcase.find("failure") is None else "Fail"
        failure_msg = testcase.find("failure").get("message", "N/A") if testcase.find("failure") is not None else "N/A"
        test_cases.append({
            "Test Case Name": name,
            "Description": f"Test in {classname}",
            "Duration (s)": f"{time:.2f}",
            "Status": status,
            "Failure Message": failure_msg
        })
    
    headers = ["Test Case Name", "Description", "Duration (s)", "Status", "Failure Message"]
    rows = [[tc[h] for h in headers] for tc in test_cases]
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    
    table = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |\n"
    table += "| " + " | ".join("-" * w for w in col_widths) + " |\n"
    for row in rows:
        table += "| " + " | ".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths)) + " |\n"
    
    with open(output_file, "w") as f:
        f.write("# Test Case Summary\n\n")
        f.write(f"Total Tests: {len(test_cases)}\n")
        f.write(f"Passed: {sum(1 for tc in test_cases if tc['Status'] == 'Pass')}\n")
        f.write(f"Failed: {sum(1 for tc in test_cases if tc['Status'] == 'Fail')}\n\n")
        f.write(table)

if __name__ == "__main__":
    generate_summary("test-results.xml", "test_summary.md")