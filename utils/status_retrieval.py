import asyncio
from playwright.async_api import async_playwright
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

async def autofill_sign_in(page, email: str, password: str):
    """
    Autofill sign-in form using existing page.
    """
    logger.info("Signing in.")
    try:
        await page.wait_for_selector('#i0116', state='visible')
        await page.fill('#i0116', email)
        print(f"Filled email: {email}")
        
        await page.click('#idSIButton9')
        await page.wait_for_load_state('networkidle')
        print("Clicked Next")
        
        await page.wait_for_selector('#i0118', state='visible')
        await page.fill('#i0118', password)
        print("Filled password")
        
        await page.click('#idSIButton9')
        await page.wait_for_load_state('networkidle')
        print("Logged in")
        
        return True
            
    except Exception as e:
        print(f"Login failed: {e}")
        return False

async def grab_status(page):
    """
    Grabs status information from the current page.
    """
    logger.info("Grabbing status information.")
    try:
        await page.wait_for_load_state('networkidle')
        
        # change to day view
        await page.click('div[aria-label="Select View"]')
        await page.wait_for_selector('li[aria-label="Select View Day"]', state='visible')
        await page.click('li[aria-label="Select View Day"]')
        await page.wait_for_load_state('networkidle')

        # get all events
        events = page.locator('div[aria-label="Full day- and multiple day events"] div.sx__event[aria-label]')
        count = await events.count()
        print(f"Found {count} events")
    
        results = []
        
        # loop through them
        for i in range(count):
            event = events.nth(i)
            title = await event.locator('p.event__title').text_content()
            
            print(f"Processing: {title}")
            
            # open modal and check for updates
            await event.click()
            content_locator = page.locator('div.content[maintenances]') # check to see if it's a maintenance thing or not
            
            try:
                await content_locator.wait_for(state='visible', timeout=2000)
                has_maintenance_content = True
            except:
                has_maintenance_content = False
                print(f"No maintenance content found for: {title}")
            
            messages = []
            
            if has_maintenance_content:
                msg_elements = page.locator('div.update__message')
                for j in range(await msg_elements.count()):
                    msg = await msg_elements.nth(j).text_content()
                    if msg.strip():
                        messages.append(msg.strip())
            else:
                messages.append("Not a maintenance event.")
            
            results.append({
                'title': title.strip(),
                'messages': messages
            })
            
            # close modal
            await page.click('div.close[tabindex="0"]')
            await page.wait_for_selector('div.content[maintenances]', state='hidden')
            await asyncio.sleep(0.3)
        
        print(f"\nGot {len(results)} events")
        for r in results:
            print(f"{r['title']}: {len(r['messages'])} updates")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return []

async def scrape_session():
    """
    Fetches the status.
    """

    logger.info("Starting status retrieval session.")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            await page.goto(os.getenv("STIBO_STATUS_URL"))
            await page.wait_for_load_state('networkidle')
            
            load_dotenv()
            email = os.getenv("EMAIL")
            password = os.getenv("PASSWORD")

            print("Signing in...")
            
            await autofill_sign_in(page, email, password)
            results = await grab_status(page)
                    
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"Error while finding status events. Current network status is unavailable.")
            return None
        finally:
            print("Status retrieval session ended.")
            await browser.close()
            return results
        
def format_status_results(results):
    """
    Formats the status results into a string.
    """
    if results is None: return "Network status is currently unavailable."

    if not results: return "No current issues."
    
    issues = []
    for item in results:
        title = item.get('title', 'Unknown Issue')
        messages = item.get('messages', [])
        message_text = ' '.join(messages) if messages else 'No details available'
        issues.append(f"- {title}: {message_text}")
    
    return '\n'.join(issues)

async def main():
    await scrape_session()

if __name__ == "__main__":
    asyncio.run(main())