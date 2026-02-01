import os
import sys
import json
import csv
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    sys.exit(1)

client = genai.Client(api_key=api_key)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "Policy Suggester Backend is Running"}

# Using models discovered via check_models.py (Prioritizing Standard models for better output quality)
MODEL_CANDIDATES = [
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.5-flash", 
    "gemini-2.5-pro",
    "gemini-3.0-flash",
    "gemini-2.5-flash-lite"
]

import difflib
import io

def match_policy_in_csv(company_name, plan_name, csv_content):
    """
    Robustly matches a policy in the CSV database.
    1. Filters by exact/fuzzy company name.
    2. Uses difflib to find best plan name match.
    """
    if not plan_name or not csv_content:
        return None

    best_match = None
    highest_ratio = 0.0
    
    # Normalize inputs
    norm_company = company_name.lower().strip()
    norm_plan = plan_name.lower().strip()
    
    reader = csv.DictReader(io.StringIO(csv_content))
    
    for row in reader:
        # Check Company Match (handle "Co. Ltd" etc)
        csv_company = row.get("Insurance Company", "").lower()
        if norm_company not in csv_company and csv_company not in norm_company:
             continue # Skip if company doesn't match at all
             
        # Check Plan Match
        csv_plan = row.get("Base Plan Name", "").lower()
        ratio = difflib.SequenceMatcher(None, norm_plan, csv_plan).ratio()
        
        # Boost ratio if exact substring match
        if norm_plan in csv_plan:
            ratio += 0.1
            
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = row

    # Threshold for acceptance
    if highest_ratio > 0.5: # generous threshold due to variations
        print(f"DEBUG: Found CSV Match! Input: '{plan_name}' -> Matched: '{best_match['Base Plan Name']}' (Score: {highest_ratio:.2f})")
        return best_match
    
    return None

def parse_date(date_str):
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None

async def generate_content_with_fallback(client, contents, **kwargs):
    last_exception = None
    for model in MODEL_CANDIDATES:
        try:
            print(f"Attempting model: {model}")

            config_params = {"response_mime_type": "application/json"}
            
            # Merge kwargs into config_params (e.g. temperature)
            if kwargs:
                config_params.update(kwargs) # This allows passing temperature=0.0

            # If tools are provided, we cannot enforce JSON mime_type easily on all models
            # But the user wants JSON. 
            if "tools" in config_params:
                 # If tools are used, mime_type must be removed for some models or handled differently
                 # ideally we keep tools in config and remove mime_type if it conflicts
                 config_params.pop("response_mime_type", None)

            response = await run_in_threadpool(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_params)
            )
            print(f"Success with model: {model}")
            return response
        except Exception as e:
            print(f"Model {model} failed: {e}")
            last_exception = e
            continue
    print("All models failed.")
    raise last_exception or Exception("All models failed")

@app.post("/api/extract")
async def extract_policy(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        # Read features CSV for context
        features_csv = ""
        try:
            with open("features3.csv", "r", encoding="utf-8") as f:
                features_csv = f.read()
        except:
            features_csv = "Room Rent, NCB, Restoration, Waiting Periods, Co-pay"

        current_date_str = datetime.now().strftime("%d-%b-%Y")
        
        # Enhanced extraction prompt
        prompt = f"""Analyze this health insurance document. 
        
        REFERENCE FEATURES LIST:
        {features_csv}

        INSTRUCTIONS:
        1. Extract basic info:
           - **company**: EXTRACT THE FULL LEGAL NAME (e.g., "Go Digit General Insurance Ltd.", "HDFC ERGO General Insurance Company Ltd."). Do NOT use abbreviations like "Digit" or "HDFC".
        2. EXTRACT POLICY DATES:
           - Look for "Policy Start Date", "Inception Date", "Risk Start Date", or "Date of First Inception".
           - **CRITICAL**: Return the exact date string found (e.g., "12/05/2020") in the JSON under policy_details -> start_date.
           - Calculate "Policy Vintage" (Duration from Start Date to Current Date: {current_date_str}).
        3. EXTRACT POLICY HOLDERS: Look for names, dates of birth (DOB). **CALCULATE PRECISE AGE** based on the Current Date: {current_date_str}. Do NOT just subtract years; consider if the birthday has passed this year yet to get the exact age.
        4. EXTRACT SUM INSURED BREAKDOWN:
             - **CRITICAL**: Identify "Base Sum Insured" (A).
             - Identify "No Claim Bonus" / "Cumulative Bonus" (B).
             - **CRITICAL**: Look for "Cumulative Bonus Super" / "No Claim Bonus Super" / "Protector Shield". This is often a SEPARATE column or mentioned in the **NOTES** below the table.
             - Identify "Additional Bonus" / "Recharge" (C).
             - **CRITICAL**: Identify "Deductible" or "Aggregate Deductible" (Terms that REDUCE cover).
             - **INSTRUCTION**: READ THE NOTES section below tables carefully. If it mentions "NCB Shield" or "Super Bonus" applied, find the amount.
             - "components": Create a list of ALL distinct positive values found.
             - Labels: "Base Sum Insured", "Cumulative Bonus", "Super No Claim Bonus", "Recharge Benefit", "Deductible".
             - Example: [{{ "label": "Base Sum Insured", "value": "10,00,000" }}, {{ "label": "Deductible", "value": "50,000" }}]
             - **CRITICAL**: Do NOT include components that are Percentages (e.g. "Bonus %"). Only include the absolute currency AMOUNT.
        5. CRITICAL: Scan the document for EVERY feature listed in the "REFERENCE FEATURES LIST" above.
        6. If a feature is found, capture its specific limit, waiting period, or condition.
        7. Compile a "comprehensive_findings" text block that lists EVERY found feature and its detail.

        Return JSON:
        {{ 
          "company": "", 
          "plan": "", 
          "premium": "", 
          "coverage": "", 
          "policy_details": {{ "start_date": "", "vintage": "" }},
          "sum_insured": {{ 
             "total": "", 
             "components": [ {{ "label": "", "value": "" }} ]
          }},
          "policy_holders": [
            {{ "name": "", "dob": "", "age": "" }}
          ],
          "features_found": {{ "room_rent": "", "ncb": "", "restoration": "", "ped_wait": "", "copay": "" }},
          "comprehensive_findings": "Full text summary of all features found matched against the reference list..."
        }}
        """
        try:
            response = await generate_content_with_fallback(client, [prompt, types.Part.from_bytes(data=content, mime_type=file.content_type)], temperature=0.0)
        except Exception as api_err:
             print(f"AI Generation Error: {api_err}")
             # Pass the actual API error to the client for debugging
             raise HTTPException(status_code=500, detail=f"AI Service Error: {str(api_err)}")

        try:
            # clean json
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                 text = text.split("```")[1].split("```")[0].strip()
            
            # Auto-repair common JSON errors if needed (simple check)
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Fallback: try to find the substring between first { and last }
                try:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start != -1 and end != -1:
                        text = text[start:end]
                        data = json.loads(text)
                    else:
                        raise ValueError("No JSON found")
                except Exception:
                    print(f"FAILED TO PARSE JSON in EXTRACT. Raw text: {text}")
                    # Return safe default
                    data = {
                       "company": "Unknown", 
                       "plan": "Unknown", 
                       "premium": "0", 
                       "coverage": "0", 
                       "policy_details": { "start_date": "", "vintage": "Unknown" },
                       "sum_insured": { "total": "0", "components": [] },
                       "policy_holders": [],
                       "features_found": {},
                       "comprehensive_findings": "Could not extract data."
                    }

            # --- PYTHON SIDE: RECALCULATE AGES PRECISELY ---
            # The LLM often hallucinates the current year or does bad math. 
            # We trust the DOB extraction more than the Age calculation.
            if "policy_holders" in data and isinstance(data["policy_holders"], list):
                today = datetime.now()
                for person in data["policy_holders"]:
                    dob_str = person.get("dob", "")
                    if dob_str:
                        # Try to parse DOB
                        dob_date = parse_date(dob_str)
                        
                        if dob_date:
                            # Calculate precise age
                            age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
                            person["age"] = str(age) # Override LLM age

            # --- PYTHON SIDE: CALCULATE TOTAL SUM INSURED ---
            if "sum_insured" in data and "components" in data["sum_insured"]:
                components = data["sum_insured"]["components"]
                total_val = 0
                
                def extract_number(val_str):
                    if not val_str: return 0
                    s = str(val_str).strip().replace(',', '')
                    # Handle decimals: If there's a dot, take only the integer part
                    if '.' in s:
                        s = s.split('.')[0]
                    # Remove non-digits
                    clean = ''.join(c for c in s if c.isdigit())
                    return int(clean) if clean else 0

                def format_indian_currency(n):
                    s = str(n)
                    if len(s) <= 3:
                        return s
                    dic = s[:-3]
                    last_3 = s[-3:]
                    groups = []
                    while len(dic) > 2:
                        groups.insert(0, dic[-2:])
                        dic = dic[:-2]
                    groups.insert(0, dic)
                    return ",".join(groups) + "," + last_3

                valid_components = []
                for comp in components:
                    val = extract_number(comp.get("value", "0"))
                    label = comp.get("label", "").lower()
                    
                    if val > 0:
                        # Skip percentages from calculation
                        if "%" in label or "percent" in label:
                             pass # Do not add to total, but could add to visual list? User said "its not related here pls", so maybe exclude from visual list too.
                             # Better to just NOT add it to valid_components if we strictly want to hide it
                             continue 

                        if "deductible" in label:
                            total_val -= val
                        else:
                            total_val += val

                        comp["value"] = format_indian_currency(val) # Apply Indian Format directly
                        valid_components.append(comp) # Add only valid components

                # Update with filtered list
                data["sum_insured"]["components"] = valid_components

                # Override total
                data["sum_insured"]["total"] = format_indian_currency(total_val)

            return data

        except Exception as e:
            print(f"JSON Parse/Processing Error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare")
async def compare_policy(data: dict):
    try:
        # Read features from CSV
        features_csv = ""
        try:
            with open("features3.csv", "r", encoding="utf-8") as f:
                features_csv = f.read()
        except Exception:
            print("WARNING: features3.csv not found")

        # Read Company Tiers from CSV
        company_data_csv = ""
        try:
            with open("company_performance_ratios.csv", "r", encoding="utf-8") as f:
                company_data_csv = f.read()
        except Exception:
            print("WARNING: company_performance_ratios.csv not found")

        # Read Insurance Plans Database
        plans_database_csv = ""
        try:
            with open("Insurance_plan_dataset.csv", "r", encoding="utf-8") as f:
                plans_database_csv = f.read()
        except Exception:
            print("WARNING: Insurance_plan_dataset.csv not found")

        # Calculate Policy Tenure from Inception Date
        inception_date_str = data.get('policy_details', {}).get('start_date', '')
        calculated_tenure = "Unknown"
        
        if inception_date_str:
            inception_date = parse_date(inception_date_str)
            if inception_date:
                today = datetime.now()
                # Calculate difference in years and months
                years = today.year - inception_date.year
                months = today.month - inception_date.month
                if months < 0:
                    years -= 1
                    months += 12
                calculated_tenure = f"{years} Years {months} Months"

        # [MODIFIED] Check if company exists in CSV
        company_name = data.get("company", "").lower().strip()
        is_company_known = company_name and (company_name in company_data_csv.lower())
        
        CSV_FALLBACK_INSTRUCTION = ""
        if not is_company_known:
            print(f"Company '{company_name}' not found in CSV. Using 'Others' fallback.")
            CSV_FALLBACK_INSTRUCTION = """
            **IMPORTANT: The company name is not explicitly found in Reference Data 2.**
            You MUST use the **"Others"** row from "Ref 2 CSV DATA" for the Current Policy Stats.
            - CSR: Use the value from the "Others" row.
            - Complaints: Use the value from the "Others" row.
            - Solvency: Use the value from the "Others" row.
            - Tier: Use the value from the "Others" row.
            """

        # [NEW] Perform Strict Verification against Plan Database
        input_plan_name = data.get("policy_details", {}).get("plan", "") # Changed from "plan_name" to "plan"
        verified_row = match_policy_in_csv(data.get("company", ""), input_plan_name, plans_database_csv)
        
        VERIFIED_DATA_SECTION = ""
        if verified_row:
            # Format row as a clean string for the LLM
            row_str = " | ".join([f"{k}: {v}" for k, v in verified_row.items() if v and v != "Not Applicable"])
            VERIFIED_DATA_SECTION = f"""
            *** VERIFIED DATABASE MATCH FOR CURRENT POLICY ***
            We found an EXACT MATCH for this policy in our database:
            Match: "{verified_row.get('Base Plan Name')}" by "{verified_row.get('Insurance Company')}"
            
            OFFICIAL DATA SPECS:
            {row_str}
            
            CRITICAL INSTRUCTION: You MUST use the above 'OFFICIAL DATA SPECS' if data is missing from the PDF.
            """

        # REQUIRED PROMPT: Must match Frontend 'recommendations' schema
        prompt = f"""
        Act as an expert insurance advisor for "Share India". 
        {CSV_FALLBACK_INSTRUCTION}
        
        EXISTING POLICY DATA (Source of Truth):
        - Basic Info: {json.dumps(data)}
        - Policy Vintage (Reported): {data.get('policy_details', {}).get('vintage', 'Unknown')}
        - **CALCULATED TENURE (Use this for Waiting Period Analysis)**: {calculated_tenure}
        - Detailed Found Features: {data.get('comprehensive_findings', 'Not available')}
        
        {VERIFIED_DATA_SECTION}

        CRITICAL INSTRUCTION: If a feature is not mentioned in the existing policy, explicitly state "Not Available" instead of "Not Found".

        The existing policy company is "{data.get('company')}".
        
        REFERENCE DATA 1 (Feature Classification):
        The following CSV data contains the "Feature", "Classification" (Must Have, Good to Have, Special Features, Red Flag), and "Explanation".
        Use this EXACT classification for mapping features.
        
        Ref 1 CSV DATA:
        {features_csv}

        REFERENCE DATA 2 (Company Tiers & Performance):
        The following CSV data contains "Company Name", "Claims Paid Ratio (Higher is Better)", "Claim Repudiation Ratio (Lower is Better)", "Claims Paid Efficiency Ratio (within 3 months)", "Claims Outstanding Ratio", "Incurred Claim Ratio (ICR Health)", "Total Claims Processed", "Complaints Settlement Ratio", "Solvency Ratio" and "Tier" (1 = Best, 2 = Good, 3 = Average).
        
        Ref 2 CSV DATA:
        {company_data_csv}

        REFERENCE DATA 3 (Plan Database):
        The following CSV data contains the *ONLY* valid source for Plan Names, Premiums, Coverages, and Feature Limits.
        Do NOT invent plans. Do NOT use outside knowledge for plan details. USE THIS DATA.

        Ref 3 CSV DATA:
        {plans_database_csv}
        
        Share India offers the following Insurance Categories:
        - Life (Term, Endowment, ULIP)
        - Health (Individual, Family, Critical Illness)
        - Motor (Car, Two-Wheeler)
        - Home (Structure, Contents)
        - Travel (Domestic, International)
        - Cyber (Digital protection)
        - Personal Accident
        - Fire Insurance
        
        2. **WAITING PERIOD ANALYSIS (CRITICAL)**:
           - Use the **CALCULATED TENURE** provided above ({calculated_tenure}) as the "Time Served".
           - Compare "Time Served" vs "Waiting Period" for PEDs (Pre-Existing Diseases) and Specific Illnesses.
           - If Time Served > Waiting Period, mark the feature as "**COVERED**" or "**WAITING PERIOD OVER**".
           - If Time Served < Waiting Period, calculate precisely how many months/years are remaining.
           - Highlight this status in "Pro" (if covered) or "Con" (if still waiting).
        
        5. **COMPREHENSIVE FEATURE CHECKLIST (MANDATORY)**:
           - You MUST evaluate EVERY SINGLE FEATURE listed in "Reference Data 1" (approx 71 items) against the Current Policy.
           - **Output Format**: A JSON Array `feature_analysis`.
           - **Status Logic**:
             - "Positive": If the feature is present/covered/good.
             - "Negative": If the feature is missing, capped, or bad (e.g., Room Rent Capping, Co-pay).
           - **Value**: Short, specific finding (e.g., "Covered up to SI", "Capped at 1%", "Not Available").
           - **Ordering**: Maintain the order: Non-Negotiable -> Must Have -> Good to Have -> Special.

        6. **PRODUCT SCORE CALCULATION**:
           - Calculate a score out of 10 based on the presence of "Non-Negotiable" and "Must Have" features.
           - Formula: (Count of Positive "Non-Negotiable" & "Must Have" Features) / (Total such features) * 10.
           - Deduct points for Red Flags (e.g., -1 for Co-pay).
           - Output as `product_score` (float, e.g., 7.5).

        Compare the policy against Indian market standards.
        Output PURE JSON only following this exact schema:
        {{
            "feature_analysis": [
                {{ "category": "Non-Negotiable Benefits", "feature": "Infinite Care", "status": "Positive", "value": "Available" }},
                {{ "category": "Must Have", "feature": "Room Rent", "status": "Negative", "value": "Capped at 1%" }},
                ... (Repeat for ALL items in Ref 1 CSV) ...
            ],
            "product_score": 7.5,
            "current_policy_stats": {{
                "company": "Company Name",
                "csr": "98.5%", "csr_rank": "Top 5",
                "solvency": "1.8", "solvency_rank": "Tier 1",
                "complaints": "95%", "complaints_rank": "High"
            }},
            "recommendations": [
                {{
                    "category": "Upgrade: [Current Policy Type] (Better Coverage)",
                    "items": [
                        {{ 
                            "company": "Company Name (e.g. HDFC Ergo)",
                            "name": "Plan Name", 
                            "type": "...", 
                            "product_score": 9.2,
                            "premium": "20,000 - 25,000", 
                            "description": "...", 
                             "stats": {{ "csr": "98.5%", "solvency": "1.8", "complaints": "95%" }},
                            "benefits": ["Benefit 1", "Benefit 2", "Benefit 3"],
                            "non_negotiable": [
                                {{ "feature": "Infinite Care", "existing": "No", "proposed": "Yes", "status": "Upgrade" }}
                            ],
                            "must_have": [
                                {{ "feature": "Room Rent", "existing": "1% limit", "proposed": "No Limit", "status": "Upgrade" }}
                            ],
                            "good_to_have": [
                                {{ "feature": "Air Ambulance", "existing": "No", "proposed": "Covered up to 2.5L", "status": "Upgrade" }}
                            ],
                            "special_features": [
                                {{ "feature": "Robotic Surgery", "existing": "No", "proposed": "Covered", "status": "Upgrade" }}
                            ],
                            "red_flags": [
                                "Co-payment clause present in many competitor base plans (highlight if relevant)"
                            ]
                        }}
                    ]
                }}
            ]
        }}
        
        CRITICAL INSTRUCTION FOR 'current_policy_stats':
        - Look up the **EXISTING POLICY'S COMPANY** in "Ref 2 CSV DATA".
        - **CSR (Claim Settlement Ratio)**: Extract the value from the **"Claims Paid Ratio"** column (Col 2) and its **Rank** (Col 3).
        - **Complaints**: Extract "Complaints Settlement Ratio" (Column 9) and its **Rank** (Column 10).
        - **Solvency**: Extract "Solvency Ratio" (March 2024 value) and its **Rank** (Column 16).
        - **Format**: 
          - Ratios: Percentage or Number (e.g. "98.5%", "1.85").
          - Ranks: **ONLY THE NUMBER** (e.g. "1", "5", "10"). Do NOT add "Top" or "Tier".

        CRITICAL INSTRUCTION FOR 'stats' FIELD (Recommendations):
        - Same logic as above. Extract Ratio and Rank.
        - JSON Structure for stats: {{ "csr": "98%", "csr_rank": "5", "solvency": "1.8", "solvency_rank": "1", "complaints": "99%", "complaints_rank": "2" }}
        - **csr**: Extract "Claims Paid Ratio" from "Ref 2 CSV DATA". Format as percentage (e.g. "98.2%").
        - **solvency**: Extract "Solvency Ratio" from "Ref 2 CSV DATA". Format as number (e.g. "1.9").
        - **complaints**: Extract "Complaints Settlement Ratio" from "Ref 2". Format as percentage (e.g. "98%").
        - If data is missing for a company, estimate based on tier.

        CRITICAL INSTRUCTION FOR 'benefits' FIELD:
        - You MUST list 3-4 KEY SELLING POINTS from "Reference Data 3" (Plan Database).
        - Focus on "No Room Rent Limit", "Unlimited Restoration", "Bonus", or "No Claim Bonus".
        - Short, punchy bullet points.

        CRITICAL INSTRUCTION FOR 'status' FIELD:
        - Set "status": "Upgrade" **ONLY** if the difference is **SIGNIFICANT** and **HIGHLY BENEFICIAL** (e.g., 'No Limit' vs 'Capped', 'Covered' vs 'Not Covered', '2X Sum Insured' vs '1X').
        - Do NOT use "Upgrade" for minor differences (e.g. 5K increase in limit, small text changes). Leave status empty string "" if the difference is minor or neutral.
        - The goal is to highlight *major* wins for the user.
        
        CRITICALLY IMPORTANT INSTRUCTIONS: 
        1. **CONSISTENCY**: You MUST provide 3 recommendations. ALL 3 recommendations must have the EXACT SAME amount of detail. 
           - DO NOT truncate the list for the 2nd or 3rd recommendation. 
           - DO NOT say "Similar to above". 
           - EACH recommendation must be fully expanded.
        2. **RANKING & SELECTION (STRICT)**:
           - **PRIMARY SORT KEY**: You MUST rank recommendations by **"Claims Paid Ratio"** (High to Low).
           - **Company Tier**: Tier 1 is preferred, BUT a Tier 1 company with a LOWER Claims Paid Ratio (e.g. 86%) MUST be ranked **BELOW** a company with a HIGHER Claims Paid Ratio (e.g. 90%), even if the second one is Tier 1 or 2.
           - **Logic**: 
             1. Filter for valid companies.
             2. Sort them primarily by **Claims Paid Ratio** (Descending).
             3. Select top 3 distinct companies.
           - **Example**: If Bajaj Allianz has 90% CSR and HDFC has 86% CSR, **Bajaj Allianz MUST appear BEFORE HDFC**.

        3. **PREMIUM ESTIMATION (MANDATORY)**:
           - **CRITICAL RULE**: NEVER, under any circumstances, output "Please search" or "Check website". This is an automated report. YOU must provide the data.
           - **Method 1 (Search)**: Try to find the actual premium brochure via Google Search.
           - **Method 2 (Estimation - REQUIRED Fallback)**: If search fails, you **MUST ESTIMATE** the premium based on:
             - **Age**: {data.get('policy_holders', [{'age': 30}])[0].get('age', 30)} years
             - **Sum Insured**: {data.get('policy_details', {}).get('sum_insured', '5 Lakh')}
             - **Family Type**: {data.get('policy_type', 'Individual')}
             - **Market Knowledge**: Use your internal knowledge of 2025 Indian Health Insurance pricing.
           - **Output Format**: Always output a realistic range (e.g. "₹22,000 - ₹26,000").
           - **Constraint**: Premiums must vary by company (e.g., Star is cheaper, Niva/HDFC are premium). Do not output identical numbers for all 3.

        3. **DISPLAY RULES (STRICT)**:
           - **Company Name**: Output ONLY the official company name (e.g., "HDFC Ergo"). **DO NOT** append "(Tier 1)" or any other stats to the name string. The user wants a clean display.
           - **Description Field (CRITICAL - USP)**:
             - You MUST find the **Unique Selling Point (USP)** of this specific plan from its brochure or your knowledge.
             - START the description with "USP: [The USP]".
             - Follow it with a brief 1-line overview of why this plan is superior.
             - Example: "USP: Industry's only unlimited restoration benefit for unrelated illnesses. This plan offers..."

        4. **DATA SOURCE RULES (STRICT)**:
           - **Current Policy Data ("Existing")**:
             - PRIMARY SOURCE: Use the 'DETAILED FOUND FEATURES' text block provided at the top.
             - **SECONDARY SOURCE (CRITICAL)**: If a feature is NOT found in the text, check **Reference Data 3 (Plan Database)**. If the *Current Policy Name* matches a plan in that CSV, USE THAT DATA to fill missing fields.
             - **TERTIARY SOURCE**: If still not found, use your internal knowledge base to infer the likely value based on the policy name/company.
             - **FINAL FALLBACK**: Only output "Not Available" if the feature is completely unknown after checking PDF, CSV, and Internal Knowledge.
             - **CRITICAL**: Do NOT output "Unknown" or "Not Mentioned. If you don't see it in the text, ESTIMATE it based on standard market features for that plan.
             - **PED WAITING PERIOD LOGIC**:
               - Identify the "Pre-Existing Disease" or "PED" waiting period from the found features (usually 2, 3, or 4 years).
               - Compare it against the **Policy Vintage** ({data.get('policy_details', {}).get('vintage', 'Unknown')}).
               - If Vintage > Waiting Period, set the Existing Value to "**Passed**" or "**Waived**".
               - If Vintage < Waiting Period, set the Existing Value to "**X Years Remaining**" (calculate the difference).
           - **Recommended Policy Data ("Proposed")**: You MUST use "Reference Data 3 (Plan Database)" for ALL recommended plan details.
           - **Feature Categorization**: You MUST use "Reference Data 1" (features3.csv) to decide if a feature is "Non-Negotiable Benefits", "Must Have", "Good to Have", etc.
           - **Company Tier**: You MUST use "Reference Data 2" (Company Ratios) for Tier and reliability stats. 
           
        5. **DISPLAY RULES**:
           - **Extracted Company Name**: Output the full legal name (e.g. "The New India Assurance Co. Ltd."). 
           - **CRITICAL cleanup**: CUT OFF any text that appears **after** "Ltd." or "Co.". 
           - **REMOVE** any parenthetical text like "(Government of India Undertaking)" or "(A Joint Venture...)" if it appears after the main name.
           - Example: "The New India Assurance Co. Ltd. (Govt of India)" -> "The New India Assurance Co. Ltd."
           - **Row Visibility**: If a feature is "No" or "Unknown" for BOTH the "Existing" and "Proposed" policy, DO NOT include that row in the output JSON. We only want to see differences or relevant features.
           - **Company Tier**: You MUST use "Reference Data 2" (Company Ratios) for Tier and reliability stats.

        5. **NON-NEGOTIABLE BENEFITS (MANDATORY ALL - HIGHEST PRIORITY)**:
           - You MUST list **ALL** features categorized as "Non-Negotiable Benefits" in "Reference Data 1" (features3.csv).
           - These are critical features like "Infinite Care", "No Sub-limits", "Consumables Cover", "Inflation Protector", "No Claim Bonus", "Restoration Benefit".
           - Lookup their values in Ref 3 for the recommended plan.

        6. **MUST HAVE FEATURES (MANDATORY ALL)**: 
           - You MUST list **ALL** features categorized as "Must Have" in "Reference Data 1" (features3.csv).
           - Do NOT filter them. If there are 10, list 10. If there are 15, list 15.
           - Lookup their values in Ref 3 for the recommended plan.

        7. **GOOD TO HAVE FEATURES (MANDATORY ALL)**:
           - You MUST list **ALL** features categorized as "Good to Have" in "Reference Data 1" (features3.csv).
           - Do NOT filter them. List EVERY specific feature mentioned in that category.
           - Lookup their values in Ref 3 for the recommended plan.

        8. **SPECIAL FEATURES (STRICT LIMIT)**:
           - Select **EXACTLY 6 to 7** unique features from the "Special Features" category in "Reference Data 1".
           - Choose the most relevant/unique ones for this plan (e.g. Robot Surgery, Bariatric, etc.).
           - Do NOT list less than 6. Do NOT list more than 7.

        9. **RED FLAGS / THINGS TO AVOID**:
           - Check the "Red Flag" category in Reference Data 1 (e.g., Co-Payment, Room Rent Limits).
           - If the *Recommended Plan* has any of these, OR if the *Current Policy* has them and they are being eliminated, mention it.
           - Example logic: "The existing 20% Co-Payment is eliminated in this proposed plan." OR "Warning: This plan has a 10% Co-Payment."
           - Populate the "red_flags" JSON array with these warnings.
        10. **PROS vs CONS (STRICT FEATURE MAPPING)**:
           - **Reference**: Use 'Reference Data 1' ({features_csv}) for the list of Must Have/Good to Have features AND their "One-liner Explanation".
           - **PROS Logic**: 
             - List features from 'Reference Data 1' that the **EXISTING POLICY HAS**.
             - Format: "Feature Name: [Details/Limit]. [One-liner Explanation]"
             - Example: "Room Rent: Covered up to 1% of SI. Covers hospital room charges up to eligible limits."
             - Do NOT use prefixes like "Must Have:" or "Good to Have:".
           - **CONS Logic**: 
             - List features from 'Reference Data 1' that the **EXISTING POLICY LACKS** (or has poor limits on).
             - Format: "Feature Name: Not Covered / limit is low. [One-liner Explanation]"
             - Example: "Robotic Surgery: Not Covered. Covers advanced medical treatments like robotic surgery."
             - Do NOT use prefixes like "Missing:", "Must Have:", or "Red Flag:". Just the statement.
             - Also check "Red Flags" present in the existing policy.
           - **Format**: Return simple string arrays.
        """

        search_tool = types.Tool(google_search=types.GoogleSearch())

        try:
            response = await generate_content_with_fallback(
                client,
                contents=prompt,
                tools=[search_tool],
                temperature=0.0 # Deterministic output
            )

        except Exception as e:
             print(f"ALL MODELS FAILED: {e}")
             raise HTTPException(status_code=429, detail="All AI models are currently busy. Please try again later.")

        text = response.text
        if not text:
            raise ValueError("AI returned empty response")
        
        print(f"DEBUG: AI Raw Text (First 500 chars): {text[:500]}...")
        
        text = text.replace("```json", "").replace("```", "").strip()
        try:
             result = json.loads(text)
             print(f"DEBUG: Parsed JSON Keys: {list(result.keys())}")
             if "feature_analysis" in result:
                 print(f"DEBUG: Feature Analysis Count: {len(result['feature_analysis'])}")
                 
                 # --- DETERMINISTIC SCORE CALCULATION ---
                 # User complained about inconsistency. We calculate score in Python now.
                 try:
                     features = result['feature_analysis']
                     
                     # 1. Filter relevant categories
                     non_negotiable = [f for f in features if f.get('category') == 'Non-Negotiable Benefits']
                     must_have = [f for f in features if f.get('category') == 'Must Have']
                     
                     # 2. Count Positives
                     pos_nn = sum(1 for f in non_negotiable if f.get('status') == 'Positive')
                     pos_mh = sum(1 for f in must_have if f.get('status') == 'Positive')
                     
                     total_relevant = len(non_negotiable) + len(must_have)
                     total_positive = pos_nn + pos_mh
                     
                     # 3. Calculate Score (Matches prompt formula: (Pos / Total) * 10)
                     # We can add explicit penalty for Red Flags if valid, but let's stick to the core ratio first.
                     if total_relevant > 0:
                         calc_score = round((total_positive / total_relevant) * 10, 1)
                     else:
                         calc_score = 0.0
                         
                     # 4. Apply Deductions for Red Flags (if mapped in feature_analysis as Negative with specific keyword?)
                     # For now, let's trust the ratio.
                     
                     print(f"DEBUG: Calc Score: {calc_score} (Pos: {total_positive}/{total_relevant})")
                     result['product_score'] = calc_score
                     
                 except Exception as e:
                     print(f"DEBUG: Score Calculation Failed: {e}")
                     # Keep original or default to 0
                     if 'product_score' not in result:
                         result['product_score'] = 0.0

             if "recommendations" in result:
                 print(f"DEBUG: Recommendations Count: {len(result['recommendations'])}")
        except json.JSONDecodeError:
             # Fallback: Find the first { and last }
             try:
                 start = text.find("{")
                 end = text.rfind("}") + 1
                 if start != -1 and end != -1:
                     result = json.loads(text[start:end])
                 else:
                     raise ValueError("No JSON found in text")
             except Exception:
                 print(f"FAILED TO PARSE JSON. Raw text: {text}")
                 # Return a safe default object to prevent crash
                 result = {
                    "pros": ["Could not analyze policy details."],
                    "cons": ["AI response was not in expected format."],
                    "current_policy_stats": {
                        "company": data.get("company", "Unknown"),
                        "csr": "N/A", "csr_rank": "N/A",
                        "solvency": "N/A", "solvency_rank": "N/A",
                        "complaints": "N/A", "complaints_rank": "N/A"
                    },
                    "recommendations": []
                 }

        # Enforce cons >= pros logic (Sales Perspective)
        # Ensure pros are never more than cons to prevent the plan looking "too good" to switch from.
        if "pros" in result and "cons" in result:
             # STRICT LIMIT: Max 7 items each
            result["pros"] = result["pros"][:7]
            result["cons"] = result["cons"][:7]

            if len(result["pros"]) > len(result["cons"]):
                # Truncate pros to match the length of cons
                result["pros"] = result["pros"][:len(result["cons"])]

        # --- PYTHON SIDE: FORCE SORT BY CLAIMS PAID RATIO ---
        try:
            csr_map = {}
            with open("company_performance_ratios.csv", "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader) # Header 1
                next(reader) # Header 2
                for row in reader:
                    if len(row) > 1:
                        # Clean name: remove special chars, lowercase
                        name_key = row[0].strip().lower().replace("general insurance", "").replace("insurance", "").strip()
                        ratio_str = row[1].replace('%', '').strip()
                        try:
                            csr_map[name_key] = float(ratio_str)
                        except:
                            pass
            
            if "recommendations" in result and isinstance(result["recommendations"], list):
                def get_csr_score(rec):
                    c_name = rec.get("company", "").lower().replace("general insurance", "").replace("insurance", "").strip()
                    # Try exact match
                    if c_name in csr_map:
                        return csr_map[c_name]
                    # Try fuzzy containment
                    for k, v in csr_map.items():
                        if k in c_name or c_name in k:
                            return v
                    return 0.0
                
                # Sort Descending by CSR
                result["recommendations"].sort(key=get_csr_score, reverse=True)

        except Exception as e:
             print(f"Sorting Error: {e}") 

        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"COMPARISON ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)