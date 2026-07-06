export function render({ model, el }) {   
    const fontLink = 'https://fonts.googleapis.com/css?family=Roboto';
    const existingLink = document.querySelector(`link[href="${fontLink}"]`);
    if (!existingLink) {
      const linkTag = document.createElement('link');
      linkTag.rel = 'stylesheet';
      linkTag.href = fontLink;
      document.head.appendChild(linkTag);
    } 
    

    let button = document.querySelector('.search-button');
    function preSearch(){
        console.log("🔍 Starting preSearch...");
    
        let searchBars = document.querySelectorAll('.plusButton');
        console.log("➕ Found", searchBars.length, "plus buttons");
        searchBars.forEach((elem) =>{
            elem.click();
        });
        
        // Handle semantic search input synchronization
        let semanticSearchInput = document.querySelector('.semantic-search-box');
        if(semanticSearchInput != null){
            console.log("📝 Found semantic search with value:", semanticSearchInput.value);
            // Trigger change event to sync the value
            semanticSearchInput.dispatchEvent(new Event('change', { bubbles: true }));
            semanticSearchInput.dispatchEvent(new Event('input', { bubbles: true }));
            console.log("✅ Semantic search events dispatched");
        } else {
            console.log("❌ No semantic search input found");
        }
        
        let invisButton = document.querySelector('.hidden-button');
        if(invisButton != null) {
            console.log("🔘 Found hidden button, clicking...");
            invisButton.click();
            console.log("✅ Hidden button clicked");
        } else {
            console.log("❌ Hidden button not found!");
            // 显示所有可用的按钮来帮助调试
            let allButtons = document.querySelectorAll('button');
            console.log("🔍 Available buttons:", allButtons.length);
            allButtons.forEach((btn, index) => {
                console.log(`Button ${index}:`, btn.className, btn.textContent?.trim());
            });
        }
        
        console.log("🏁 preSearch completed");
    } 
    if(button != null){
        button.addEventListener("click", preSearch);
    }

    const query = '.date-constraint > input:first-of-type';
    let start = model.get("calendarStart");
    let end = model.get("calendarEnd");
    let results = document.querySelectorAll(query);
    results.forEach((calenderEl) => {
        calenderEl.setAttribute('min', start);
        calenderEl.setAttribute('max', end);
    });




    // Install the global confirm handlers BEFORE the file-upload decoration
    // below: that block early-returns when .widget-upload is not in the DOM
    // yet (render-order race), which used to leave window.confirmFilter and
    // friends undefined ("window.confirmFilter is not a function").
    function stanceRunning(){
        return model.get("activeStanceAnalysis") == 1;
    }
    function confirmChanges(){
        let response = confirm("Apply filters?\nYou have made changes to the filter settings, but they have not been applied yet.");
        let responseCode = 0;
        if(response){
            if(stanceRunning()){
                let comfirmation = confirm("Apply Refine Results modification?\nStance Annotation is running in the background. Modifying the Refine Results selections will end Stance Annotation. If you want to continue Refining Results, click OK.");
                if(comfirmation){
                    responseCode = 1;
                }
            }
            else{
                responseCode = 1;
            }
        }
        model.set("userResponse", responseCode);
        model.set("changeSignal", model.get("changeSignal") + 1);
        model.save_changes();
    }
    function doComfirmation(prompt, varName){
        let responseCode = 1;
        if(stanceRunning()){
            let response = confirm(prompt);
            if(!response){
                responseCode = 0;
            }
        }
        model.set("userResponse", responseCode);
        model.set(varName, model.get(varName) + 1);
        model.save_changes();
    }
    window.confirmChanges = confirmChanges;
    window.confirmSearch = function(){
        let prompt = "Attempt to modify Search Criteria?\nStance Annotation is running in the background. Modifying Search Criteria will end Stance Annotation. If you want to continue modifying Search Criteria, click OK.";
        doComfirmation(prompt, "searchChangeSignal");
    }
    window.confirmFilter = function(){
        let prompt = "Apply Refine Results modification?\nStance Annotation is running in the background. Modifying the Refine Results selections will end Stance Annotation. If you want to continue Refining Results, click OK.";
        doComfirmation(prompt, "filterChangeSignal");
    }
    if(!window.__dummyAlertWired){
        window.__dummyAlertWired = true;
        // dummyEl can be re-rendered on every resetDisplay; wire the shared
        // model/window listeners once and route through window.* so they
        // always hit the latest render's closures.
        model.on("change:alertTrigger", function(){ window.confirmChanges(); });
        window.addEventListener("beforeunload", (event) => {
            if (stanceRunning()) {
                event.preventDefault();
                event.returnValue = "";
            }
        });
    }

    let fileUp = document.querySelector(".widget-upload");
    if(fileUp == null){
        return;
    }
    if(document.getElementById("file-upload-cover") != null){
        return;
    }
    fileUp.setAttribute("id", "file-upload");
    let parent = fileUp.parentElement;

    let label = document.createElement("label");
    label.classList.add("dataset-display");     
    label.setAttribute("id", "file-upload-cover"); 

    label.htmlFor = "file-upload";

    let leftText = document.createElement("div");

    let fileName = model.get("fileName");
    if(fileName == null || fileName == ""){
        fileName = "No File Loaded";
    }
    let filePath = model.get("filePath");
    let fileIcon = document.createElement("img");
    fileIcon.src = filePath + "file.svg";

    let uploadIcon = document.createElement("img");
    uploadIcon.src = filePath + "upload.svg";
    
    leftText.innerHTML = "&nbsp; &nbsp; <b><u>" + fileName + "</u></b> &nbsp; " + model.get("size").toLocaleString() + " Posts &nbsp; &nbsp;";



    label.appendChild(fileIcon);
    label.appendChild(leftText);
    label.appendChild(uploadIcon);
    parent.appendChild(label);

    // function shutdownKernel(){ // from .local/share/jupyter/voila/templates/base/static/main.js
    //     const matches = document.cookie.match('\\b_xsrf=([^;]*)\\b');
    //     const xsrfToken = (matches && matches[1]) || '';
    //     const configData = JSON.parse(document.getElementById('jupyter-config-data').textContent);
    //     const baseUrl = configData.baseUrl;
    //     const data = new FormData();
    //     data.append("_xsrf", xsrfToken);
    //     window.navigator.sendBeacon(`${baseUrl}voila/api/shutdown/${kernel.id}`, data);
    //     // kernel.dispose();
    // }
}