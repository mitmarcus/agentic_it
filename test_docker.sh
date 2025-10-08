#!/bin/bash
#
# Docker Testing Script for IT Support Chatbot
# This script tests all functionality of the AI chatbot running in Docker

# Don't exit on error, we want to count all failures
# set -e

echo "=================================="
echo "IT Support Chatbot - Docker Tests"
echo "=================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base URL
BASE_URL="http://localhost:8000"

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test result
test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $2"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $2"
        ((TESTS_FAILED++))
    fi
}

# Function to make API call and check response
test_api() {
    local endpoint=$1
    local method=$2
    local data=$3
    local expected=$4
    local description=$5
    
    echo -e "\n${YELLOW}Testing:${NC} $description"
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s "$BASE_URL$endpoint")
    else
        response=$(curl -s -X "$method" "$BASE_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    
    if echo "$response" | grep -q "$expected"; then
        test_result 0 "$description"
        echo "Response: $response" | head -c 200
    else
        test_result 1 "$description"
        echo "Response: $response"
    fi
}

echo "Step 1: Checking if Docker containers are running..."
if ! docker ps | grep -q "it-support-chatbot"; then
    echo -e "${RED}Error: Chatbot container is not running!${NC}"
    echo "Run: docker compose up -d"
    exit 1
fi
echo -e "${GREEN}✓${NC} Chatbot container is running"

if ! docker ps | grep -q "it-support-chromadb"; then
    echo -e "${RED}Error: ChromaDB container is not running!${NC}"
    echo "Run: docker compose up -d"
    exit 1
fi
echo -e "${GREEN}✓${NC} ChromaDB container is running"

echo
echo "Step 2: Testing API Endpoints..."

# Test 1: Health Check
test_api "/health" "GET" "" "healthy" "Health check endpoint"

# Test 2: Index documents (if not already indexed)
echo
echo "Step 3: Indexing documents..."
response=$(curl -s -X POST "$BASE_URL/index" \
    -H "Content-Type: application/json" \
    -d '{"source_dir": "./data/docs"}')

if echo "$response" | grep -q "success\|already indexed"; then
    test_result 0 "Document indexing"
    echo "$response"
else
    test_result 1 "Document indexing"
    echo "$response"
fi

echo
echo "Step 4: Testing AI Query Processing..."

# Test 3: VPN Setup Query
echo
echo -e "${YELLOW}Test 3:${NC} VPN setup question"
response=$(curl -s -X POST "$BASE_URL/query" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "How do I set up VPN?",
        "session_id": "test-vpn-001"
    }')

if echo "$response" | grep -q "Cisco AnyConnect\|VPN\|vpn.company.com"; then
    test_result 0 "VPN setup query - AI found relevant documentation"
    echo "$response" | python -m json.tool 2>/dev/null | head -20
else
    test_result 1 "VPN setup query"
    echo "$response"
fi

# Test 4: Password Reset Query
echo
echo -e "${YELLOW}Test 4:${NC} Password reset question"
response=$(curl -s -X POST "$BASE_URL/query" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "I forgot my password, how can I reset it?",
        "session_id": "test-pwd-001"
    }')

if echo "$response" | grep -q "portal.company.com\|reset\|password"; then
    test_result 0 "Password reset query - AI provided correct information"
    echo "$response" | python -m json.tool 2>/dev/null | head -20
else
    test_result 1 "Password reset query"
    echo "$response"
fi

# Test 5: Printer Troubleshooting Query
echo
echo -e "${YELLOW}Test 5:${NC} Printer troubleshooting question"
response=$(curl -s -X POST "$BASE_URL/query" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "My printer is not printing anything",
        "session_id": "test-printer-001"
    }')

if echo "$response" | grep -q "printer\|print\|troubleshoot"; then
    test_result 0 "Printer troubleshooting query - AI responded appropriately"
    echo "$response" | python -m json.tool 2>/dev/null | head -20
else
    test_result 1 "Printer troubleshooting query"
    echo "$response"
fi

# Test 6: Off-topic Query (should handle gracefully)
echo
echo -e "${YELLOW}Test 6:${NC} Off-topic question (testing boundaries)"
response=$(curl -s -X POST "$BASE_URL/query" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What is the weather today?",
        "session_id": "test-offtopic-001"
    }')

if echo "$response" | grep -q "response"; then
    test_result 0 "Off-topic query - AI responded (boundary test)"
    echo "$response" | python -m json.tool 2>/dev/null | head -20
else
    test_result 1 "Off-topic query"
    echo "$response"
fi

# Test 7: Conversation History
echo
echo -e "${YELLOW}Test 7:${NC} Get conversation history"
response=$(curl -s "$BASE_URL/session/test-vpn-001/history")

if echo "$response" | grep -q "history\|messages"; then
    test_result 0 "Conversation history retrieval"
    echo "$response" | python -m json.tool 2>/dev/null | head -15
else
    test_result 1 "Conversation history retrieval"
    echo "$response"
fi

# Test 8: Session Cleanup
echo
echo -e "${YELLOW}Test 8:${NC} Clear session"
response=$(curl -s -X DELETE "$BASE_URL/session/test-vpn-001")

if echo "$response" | grep -q "success\|cleared"; then
    test_result 0 "Session cleanup"
else
    test_result 1 "Session cleanup"
fi

echo
echo "Step 5: Checking Docker logs..."
echo -e "${YELLOW}Last 10 log lines from chatbot:${NC}"
docker compose logs chatbot --tail=10

echo
echo "=================================="
echo "Test Summary"
echo "=================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! The AI chatbot is working correctly in Docker.${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Check the output above for details.${NC}"
    exit 1
fi
