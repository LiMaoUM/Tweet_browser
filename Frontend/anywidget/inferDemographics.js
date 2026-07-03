function createAndAdd(element, className, parent) {
    const el = document.createElement(element);
    el.classList.add(className);
    parent.appendChild(el);
    return el;
}

export function render({ model, el }) {
    // Add main class to container
    el.classList.add("infer-demographics");
    
    // Create info page
    let infoPage = createAndAdd(el, "", "info-page");
    createAndAdd(infoPage, "Infer Demographics", "heading4").classList.add("medium");
    // createAndAdd(infoPage, "Demographic Analysis", "");
    createAndAdd(infoPage, "Infer Demographics initiates machine-based modeling to infer eight distinct demographic attributes (“inferred attributes”) for each social media user who posted any content within a single set of sampled posts. The process may take a few minutes and will be quicker with smaller sample sizes (a sample of fewer than 1000 posts is recommended). To adjust the sample size, update Searched Criteria and/or further refine results, as necessary. For example, narrowing the date range could decrease the sample size. <br>These attributes (sex, age, education, urbanicity, income, partisanship, political ideology, metro area [or not]) are predictions based on our model, not verified or provided by the user. Accuracy may vary by dataset. Please take this into account when using the inferred demographics. <br>In our benchmark tests based on social media posters' self-reported attributes in one dataset, the model was most accurate for sex (~90%, inferring two categories), age (~60%, inferring four categories), and education (~60%, inferring three categories); the other attributes had lower accuracy, but all were above chance. Again, accuracy in different datasets may vary.");
    
    let nextButton = document.createElement("button");
    nextButton.innerHTML = "Start Demographic Analysis";
    nextButton.classList.add("generic-button");
    nextButton.addEventListener("click", dismissInfo);
    infoPage.appendChild(nextButton);

    function createAndAdd(parent, html, cssClass){
        let temp = document.createElement("div");
        temp.innerHTML = html;
        if(cssClass != ""){
            temp.classList.add(cssClass);
        }
        parent.appendChild(temp);
        return temp;
    }

    function dismissInfo(){
        // Trigger page change
        model.set("pageNumber", 1);
        model.save_changes();
    }

    return el;
}
