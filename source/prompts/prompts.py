


PROMPT_TEMPLATE = """
Act as an Automatic Program Repair (APR) tool, reply only with code, without explanation. 
You are specialized in breaking dependency updates, in which the failure is caused by an external dependency. 
To solve the failure you can only work on the client code.

the following client code fails: 
'''java
{client_code}
'''
the error is triggered in the following specific lines in the previous code:
{buggy_line}
with the following error message:
{error_message}
The error is caused by a change in the API of the dependency. The new library version includes the following changes: <api diff>
{api_diff}
Before proposing a fix, please analyze the situation and plan your approach within
<repair strategy> tags:

- Identify the specific API changes that are causing the failure in the client code. 
- Compare the old and new API versions, noting any changes in method signatures, return types, or parameter lists. 
- Determine which parts of the client code need to be updated to accommodate these API changes. 
- Consider any constraints or requirements for the fix (e.g., not changing function signatures, potential import adjustments). 
- Plan the minimal set of changes needed to fix the issue while keeping the code functional and compliant with the new API. 
- Consider potential side effects of the proposed changes on other parts of the code. 
- Ensure that the planned changes will result in a complete and compilable class. 
- If applicable, note any additional imports that may be needed due to the API changes.  
- Propose a patch that can be applied to the code to fix the issue. 
- Return only a complete and compilable class in a fenced code block. 
- You CANNOT change the function signature of any method but may create variables if it simplifies the code. 
- You CAN remove the @Override annotation IF AND ONLY IF the method no longer overrides a method in the updated dependency version. 
- If fixing the issue requires addressing missing imports, ensure the correct package or class is used in accordance with the newer dependency version. 
- Avoid removing any existing code unless it directly causes a compilation or functionality error. 
- Return only the fixed class, ensuring it fully compiles and adheres to these constraints.
"""