// Expose global functions for popup and modal actions
window.selectStyle = selectStyle;
window.closeModal = closeModal;

// Global variable to hold the last search query
let lastQuery = "";

/**
 * Close the diamond details modal
 */
function closeModal() {
  document.getElementById("detailsModal").style.display = "none";
}

/**
 * Remove <diamond-data> JSON from the text and bold only the *values* in <expert-analysis>.
 */
function processChatMessage(text) {
  // 1) Remove <diamond-data> block
  let processed = text.replace(/<diamond-data>[\s\S]*?<\/diamond-data>/, "");
  
  // 2) If there's <expert-analysis>, bold only the numeric/letter values after attributes
  processed = processed.replace(/<expert-analysis>([\s\S]*?)<\/expert-analysis>/, (m, p1) => {
    return boldValues(p1);
  });
  
  return processed;
}

/**
 * Bold only the values after attribute labels (Carat, Clarity, etc.).
 */
function boldValues(text) {
  let newText = text;
  newText = newText.replace(/(Carat:\s*)([^\s,]+)/gi, '$1<strong>$2</strong>');
  newText = newText.replace(/(Clarity:\s*)([^\s,]+)/gi, '$1<strong>$2</strong>');
  newText = newText.replace(/(Color:\s*)([^\s,]+)/gi, '$1<strong>$2</strong>');
  newText = newText.replace(/(Cut:\s*)([^\s,]+)/gi, '$1<strong>$2</strong>');
  newText = newText.replace(/(Polish:\s*)([^\s,]+)/gi, '$1<strong>$2</strong>');
  newText = newText.replace(/(Symmetry:\s*)([^\s,]+)/gi, '$1<strong>$2</strong>');
  return newText;
}

/**
 * Add a chat message bubble. Bot on left, user on right, with small circular avatars.
 */
function addMessage(message, isUser = false) {
  const chatMessages = document.getElementById("chatMessages");
  
  // Create message container
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message");
  if (isUser) {
    msgDiv.classList.add("user-message");
  } else {
    msgDiv.classList.add("bot-message");
  }
  
  // Create the avatar container
  const avatarDiv = document.createElement("div");
  avatarDiv.classList.add("avatar");
  if (isUser) {
    avatarDiv.classList.add("user-avatar");
  } else {
    avatarDiv.classList.add("bot-avatar");
  }
  
  // Create avatar icon using FontAwesome classes
  const avatarIcon = document.createElement("i");
  if (isUser) {
    avatarIcon.classList.add("fas", "fa-user");
    avatarIcon.setAttribute("aria-label", "User Avatar");
  } else {
    avatarIcon.classList.add("fas", "fa-gem");
    avatarIcon.setAttribute("aria-label", "Gemma Avatar");
  }
  avatarDiv.appendChild(avatarIcon);
  
  // Create the bubble for message content
  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  if (isUser) {
    bubble.classList.add("user-bubble");
  } else {
    bubble.classList.add("bot-bubble");
  }
  
  bubble.innerHTML = processChatMessage(message);
  
  // Append in correct order based on whether message is from user or bot
  if (isUser) {
    msgDiv.appendChild(bubble);
    msgDiv.appendChild(avatarDiv);
  } else {
    msgDiv.appendChild(avatarDiv);
    msgDiv.appendChild(bubble);
  }
  
  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Update the search results bar text.
 */
function updateSearchResultsBar(text) {
  const searchResultsBar = document.getElementById("search-results-bar");
  if (searchResultsBar) {
    searchResultsBar.textContent = text;
  }
}

/**
 * Show the style popup.
 */
function showStylePopup() {
  document.getElementById("stylePopup").style.display = "flex";
}

/**
 * Called when user picks labgrown or natural.
 */
function selectStyle(style) {
  document.getElementById("stylePopup").style.display = "none";
  const userInput = document.getElementById("userInput");
  const sendButton = document.getElementById("sendButton");
  
  // Retrieve pending query
  const pendingQuery = localStorage.getItem("pendingQuery");
  if (pendingQuery) {
    localStorage.removeItem("pendingQuery");
    const newQuery = `${pendingQuery} ${style}`;
    addMessage(`Style preference: ${style}`, true);
    
    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: newQuery })
    })
    .then(res => res.json())
    .then(data => {
      addMessage(data.response, false);
      handleDiamondData(data.response);
    })
    .catch(err => {
      console.error("Error:", err);
      addMessage("Sorry, there was an error processing your request.", false);
    });
  }
  
  userInput.disabled = false;
  sendButton.disabled = false;
}

/**
 * Handle diamond data from <diamond-data> and display in a grid, with expert analysis above the cards.
 */
function handleDiamondData(responseText) {
  const diamondDataRegex = /<diamond-data>([\s\S]*?)<\/diamond-data>/;
  const match = responseText.match(diamondDataRegex);
  
  const expertAnalysisRegex = /<expert-analysis>([\s\S]*?)<\/expert-analysis>/;
  const analysisMatch = responseText.match(expertAnalysisRegex);
  let expertAnalysis = analysisMatch ? analysisMatch[1] : null;
  
  const dynamicResults = document.getElementById("dynamic-results");
  dynamicResults.innerHTML = ""; // Clear old results
  
  // If there's expert analysis, show it in a separate card above
  if (expertAnalysis) {
    expertAnalysis = boldValues(expertAnalysis);
    const analysisCard = document.createElement("div");
    analysisCard.classList.add("expert-analysis-card");
    analysisCard.innerHTML = `
      <h3><i class="fas fa-gem"></i> Expert Recommendation</h3>
      <div class="expert-analysis-content">${expertAnalysis}</div>
    `;
    dynamicResults.appendChild(analysisCard);
  }
  
  if (match && match[1]) {
    try {
      const diamondData = JSON.parse(match[1]);
      
      // Update search results bar with count information
      updateSearchResultsBar(`Found ${diamondData.length} Diamond${diamondData.length !== 1 ? 's' : ''} for '${lastQuery}'`);
      
      // Create a container for the cards
      const resultsGrid = document.createElement("div");
      resultsGrid.classList.add("diamond-results");
      
      // Create a card for each diamond
      diamondData.forEach((diamond) => {
        const card = document.createElement("div");
        card.classList.add("diamond-card");
        
        // Header
        const headerDiv = document.createElement("div");
        headerDiv.className = "diamond-card-header";
        
        const iconDiv = document.createElement("div");
        iconDiv.className = "diamond-icon";
        iconDiv.innerHTML = '<i class="fas fa-gem"></i>';
        
        const titleDiv = document.createElement("div");
        titleDiv.className = "diamond-title";
        
        const caratDiv = document.createElement("div");
        caratDiv.className = "diamond-carat";
        caratDiv.textContent = `${diamond.Carat} Carat`;
        
        const shapeDiv = document.createElement("div");
        shapeDiv.className = "diamond-shape";
        shapeDiv.textContent = diamond.Shape;
        
        titleDiv.appendChild(caratDiv);
        titleDiv.appendChild(shapeDiv);
        
        headerDiv.appendChild(iconDiv);
        headerDiv.appendChild(titleDiv);
        
        // Specs
        const specsDiv = document.createElement("div");
        specsDiv.className = "diamond-specs";
        
        const specs = [
          { label: "Clarity", value: diamond.Clarity },
          { label: "Color", value: diamond.Color },
          { label: "Cut", value: diamond.Cut },
          { label: "Polish", value: diamond.Polish },
          { label: "Symmetry", value: diamond.Symmetry },
          { label: "Style", value: diamond.Style },
        ];
        
        specs.forEach(spec => {
          const specRow = document.createElement("div");
          specRow.className = "diamond-spec";
          
          const labelSpan = document.createElement("div");
          labelSpan.className = "spec-label";
          labelSpan.textContent = `${spec.label}:`;
          
          const valueSpan = document.createElement("div");
          valueSpan.className = "spec-value";
          valueSpan.textContent = spec.value || "N/A";
          
          specRow.appendChild(labelSpan);
          specRow.appendChild(valueSpan);
          specsDiv.appendChild(specRow);
        });
        
        // Price
        const priceDiv = document.createElement("div");
        priceDiv.className = "diamond-price";
        const priceVal = document.createElement("div");
        priceVal.className = "price-value";
        
        let formattedPrice;
        try {
          formattedPrice = new Intl.NumberFormat("en-US", {
            style: "currency",
            currency: "USD",
            maximumFractionDigits: 0
          }).format(diamond.Price);
        } catch (e) {
          formattedPrice = `$${diamond.Price}`;
        }
        priceVal.textContent = formattedPrice;
        priceDiv.appendChild(priceVal);
        
        // View details button
        const detailsBtn = document.createElement("button");
        detailsBtn.className = "view-details-btn";
        detailsBtn.textContent = "View Details";
        detailsBtn.onclick = function() {
          openModalWithDetails(diamond);
        };
        
        // Build the card
        card.appendChild(headerDiv);
        card.appendChild(specsDiv);
        card.appendChild(priceDiv);
        card.appendChild(detailsBtn);
        
        resultsGrid.appendChild(card);
      });
      
      dynamicResults.appendChild(resultsGrid);
      
    } catch (err) {
      console.error("Error parsing diamond data:", err);
    }
  } else {
    // If no diamond data is found, clear the search results bar
    updateSearchResultsBar("");
  }
}

/**
 * Open a modal with diamond details.
 */
function openModalWithDetails(diamond) {
  const modal = document.getElementById("detailsModal");
  const modalDetails = document.getElementById("modalDetails");
  modalDetails.innerHTML = `
    <h2>${diamond.Carat} Carat Diamond</h2>
    <p><strong>Clarity:</strong> ${diamond.Clarity}</p>
    <p><strong>Color:</strong> ${diamond.Color}</p>
    <p><strong>Cut:</strong> ${diamond.Cut}</p>
    <p><strong>Polish:</strong> ${diamond.Polish}</p>
    <p><strong>Symmetry:</strong> ${diamond.Symmetry}</p>
    <p><strong>Style:</strong> ${diamond.Style}</p>
    <p><strong>Price:</strong> $${diamond.Price}</p>
  `;
  modal.style.display = "flex";
}

/**
 * Send user message to server.
 */
function sendMessage() {
  const input = document.getElementById("userInput");
  const message = input.value.trim();
  if (!message) return;
  
  // Update the global query variable and search results bar
  lastQuery = message;
  updateSearchResultsBar(`Searching for '${message}'.....`);
  
  // Add user message to chat
  addMessage(message, true);
  input.value = "";
  
  fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message })
  })
  .then(res => res.json())
  .then(data => {
    if (data.needs_style) {
      localStorage.setItem("pendingQuery", message);
      showStylePopup();
      return;
    }
    addMessage(data.response, false);
    handleDiamondData(data.response);
  })
  .catch(err => {
    console.error("Error:", err);
    addMessage("Sorry, I encountered an error processing your request.", false);
  });
}

// Attach event listeners for the send button and Enter key
document.addEventListener("DOMContentLoaded", function () {
  const sendButton = document.getElementById("sendButton");
  const userInput = document.getElementById("userInput");
  
  sendButton.addEventListener("click", sendMessage);
  userInput.addEventListener("keydown", function(e) {
    if (e.key === "Enter") {
      sendMessage();
    }
  });
});
