import json
import os
import logging
from jinja2 import Template

# Initialize logger for this module
logger = logging.getLogger(__name__)


class ReportTool:
    def __init__(self, output_folder="output"):
        """
        Initialize the report generator and ensure the output directory exists.
        """
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info(f"Created output directory: {output_folder}")

    def write_html_report(self, rules_input):
        """
        Parses agent input and renders an HTML report using the Jinja2 template engine.
        """
        logger.info("Starting HTML report generation process.")

        # Parse input: handle both raw string JSON from LLM and Python objects
        if isinstance(rules_input, str):
            try:
                logger.debug("Input is a string. Attempting to clean and parse JSON.")
                # Remove Markdown formatting if present
                clean_input = rules_input.replace("```json", "").replace("```", "").strip()
                rules_data = json.loads(clean_input)
                logger.info(f"Successfully parsed JSON string containing {len(rules_data)} rules.")
            except Exception as e:
                logger.error(f"JSON parsing failed: {e}")
                return f"Error parsing JSON: {str(e)}"
        else:
            rules_data = rules_input
            logger.info(f"Received input as Python list with {len(rules_data)} rules.")

        # Define the visual structure and styling of the final audit report
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: sans-serif; margin: 40px; background: #f4f7f6; }
                .rule-card { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-left: 5px solid #2c3e50; }
                pre { background: #272822; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; }
                .tag { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; text-transform: uppercase; }
                .High { background: #d4edda; color: #155724; }
                .Mutual { border-left-color: #e74c3c; }
            </style>
        </head>
        <body>
            <h1>Medical Policy Analysis - SQL Rules</h1>
            {% for rule in rules %}
            <div class="rule-card {{ rule.classification.split()[0] }}">
                <h2>{{ rule.name }}</h2>
                <span class="tag {{ rule.confidence }}">{{ rule.confidence }} Confidence</span>
                <p><strong>Type:</strong> {{ rule.classification }}</p>
                <p><strong>Description:</strong> {{ rule.description }}</p>
                <pre>{{ rule.sql }}</pre>
                <p style="color: #666; font-style: italic;">"{{ rule.quote }}"</p>
            </div>
            {% endfor %}
        </body>
        </html>
        """

        try:
            # Render data into the HTML template
            logger.debug("Rendering HTML with Jinja2 template.")
            template = Template(html_template)
            html_content = template.render(rules=rules_data)

            # Save the rendered HTML to the output folder
            file_path = os.path.join(self.output_folder, "final_report.html")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"Report successfully saved to: {file_path}")
            return f"Success! Report generated with {len(rules_data)} rules."

        except Exception as e:
            logger.error(f"Failed to render or save HTML report: {e}", exc_info=True)
            return f"Error during report rendering: {str(e)}"