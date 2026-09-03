console.log("script.js chargé");


// =========================================================
// EXPERIMENT PARAMETERS
// =========================================================

const IMAGE_INSTRUCTION_DURATION_MS = 3000;

const IMAGE_PRESENTATION_DURATION_MS = 2000;

const MATRIX_SIZE = 64;

const MATRIX_ROWS = 8;

const MATRIX_COLUMNS = 8;

const TARGET_COUNTS = [
    8,
    14,
    20,
    26,
    32,
    38,
    44,
    50,
    56
];


// =========================================================
// IMAGE TRIALS
// =========================================================
// =========================================================
// IMAGE TRIALS
// =========================================================

const imageTrials = [

    // =====================================================
    // 8 / 64 = 12.5 %
    // =====================================================

    {
        id: "8_blue_green",

        target_count: 8,
        total_count: 64,
        true_percentage: 100 * 8 / 64,

        image_type: "circles_blue_green",

        target_group: "blue",

        question:
            "Quel pourcentage des cercles étaient bleus ?",

        image:
            "images_matrices/8of64/matrix_circles_blue_green.jpg"
    },

    {
        id: "8_green_blue",

        target_count: 8,
        total_count: 64,
        true_percentage: 100 * 8 / 64,

        image_type: "circles_green_blue",

        target_group: "green",

        question:
            "Quel pourcentage des cercles étaient verts ?",

        image:
            "images_matrices/8of64/matrix_circles_green_blue.jpg"
    },

    {
        id: "8_face",

        target_count: 8,
        total_count: 64,
        true_percentage: 100 * 8 / 64,

        image_type: "face",

        target_group: "black",

        question:
            "Quel pourcentage des personnes étaient noires ?",

        image:
            "images_matrices/8of64/matrix_face.jpg"
    },


    // =====================================================
    // 14 / 64 = 21.875 %
    // =====================================================

    {
        id: "14_blue_green",

        target_count: 14,
        total_count: 64,
        true_percentage: 100 * 14 / 64,

        image_type: "circles_blue_green",

        target_group: "blue",

        question:
            "Quel pourcentage des cercles étaient bleus ?",

        image:
            "images_matrices/14of64/matrix_circles_blue_green.jpg"
    },

    {
        id: "14_green_blue",

        target_count: 14,
        total_count: 64,
        true_percentage: 100 * 14 / 64,

        image_type: "circles_green_blue",

        target_group: "green",

        question:
            "Quel pourcentage des cercles étaient verts ?",

        image:
            "images_matrices/14of64/matrix_circles_green_blue.jpg"
    },

    {
        id: "14_face",

        target_count: 14,
        total_count: 64,
        true_percentage: 100 * 14 / 64,

        image_type: "face",

        target_group: "black",

        question:
            "Quel pourcentage des personnes étaient noires ?",

        image:
            "images_matrices/14of64/matrix_face.jpg"
    },


    // =====================================================
    // 20 / 64 = 31.25 %
    // =====================================================

    {
        id: "20_blue_green",

        target_count: 20,
        total_count: 64,
        true_percentage: 100 * 20 / 64,

        image_type: "circles_blue_green",

        target_group: "blue",

        question:
            "Quel pourcentage des cercles étaient bleus ?",

        image:
            "images_matrices/20of64/matrix_circles_blue_green.jpg"
    },

    {
        id: "20_green_blue",

        target_count: 20,
        total_count: 64,
        true_percentage: 100 * 20 / 64,

        image_type: "circles_green_blue",

        target_group: "green",

        question:
            "Quel pourcentage des cercles étaient verts ?",

        image:
            "images_matrices/20of64/matrix_circles_green_blue.jpg"
    },

    {
        id: "20_face",

        target_count: 20,
        total_count: 64,
        true_percentage: 100 * 20 / 64,

        image_type: "face",

        target_group: "black",

        question:
            "Quel pourcentage des personnes étaient noires ?",

        image:
            "images_matrices/20of64/matrix_face.jpg"
    },


    // =====================================================
    // 26 / 64 = 40.625 %
    // =====================================================

    {
        id: "26_blue_green",

        target_count: 26,
        total_count: 64,
        true_percentage: 100 * 26 / 64,

        image_type: "circles_blue_green",

        target_group: "blue",

        question:
            "Quel pourcentage des cercles étaient bleus ?",

        image:
            "images_matrices/26of64/matrix_circles_blue_green.jpg"
    },

    {
        id: "26_green_blue",

        target_count: 26,
        total_count: 64,
        true_percentage: 100 * 26 / 64,

        image_type: "circles_green_blue",

        target_group: "green",

        question:
            "Quel pourcentage des cercles étaient verts ?",

        image:
            "images_matrices/26of64/matrix_circles_green_blue.jpg"
    },

    {
        id: "26_face",

        target_count: 26,
        total_count: 64,
        true_percentage: 100 * 26 / 64,

        image_type: "face",

        target_group: "black",

        question:
            "Quel pourcentage des personnes étaient noires ?",

        image:
            "images_matrices/26of64/matrix_face.jpg"
    },


    // =====================================================
    // 32 / 64 = 50 %
    // =====================================================

    {
        id: "32_blue_green",

        target_count: 32,
        total_count: 64,
        true_percentage: 100 * 32 / 64,

        image_type: "circles_blue_green",

        target_group: "blue",

        question:
            "Quel pourcentage des cercles étaient bleus ?",

        image:
            "images_matrices/32of64/matrix_circles_blue_green.jpg"
    },

    {
        id: "32_green_blue",

        target_count: 32,
        total_count: 64,
        true_percentage: 100 * 32 / 64,

        image_type: "circles_green_blue",

        target_group: "green",

        question:
            "Quel pourcentage des cercles étaient verts ?",

        image:
            "images_matrices/32of64/matrix_circles_green_blue.jpg"
    },

    {
        id: "32_face",

        target_count: 32,
        total_count: 64,
        true_percentage: 100 * 32 / 64,

        image_type: "face",

        target_group: "black",

        question:
            "Quel pourcentage des personnes étaient noires ?",

        image:
            "images_matrices/32of64/matrix_face.jpg"
    },


    // =====================================================
    // 38 / 64 = 59.375 %
    // =====================================================

    {
        id: "38_blue_green",

        target_count: 38,
        total_count: 64,
        true_percentage: 100 * 38 / 64,

        image_type: "circles_blue_green",

        target_group: "blue",

        question:
            "Quel pourcentage des cercles étaient bleus ?",

        image:
            "images_matrices/38of64/matrix_circles_blue_green.jpg"
    },

    {
        id: "38_green_blue",

        target_count: 38,
        total_count: 64,
        true_percentage: 100 * 38 / 64,

        image_type: "circles_green_blue",

        target_group: "green",

        question:
            "Quel pourcentage des cercles étaient verts ?",

        image:
            "images_matrices/38of64/matrix_circles_green_blue.jpg"
    },

    {
        id: "38_face",

        target_count: 38,
        total_count: 64,
        true_percentage: 100 * 38 / 64,

        image_type: "face",

        target_group: "black",

        question:
            "Quel pourcentage des personnes étaient noires ?",

        image:
            "images_matrices/38of64/matrix_face.jpg"
    },


    // =====================================================
    // 44 / 64 = 68.75 %
    // =====================================================

    {
        id: "44_blue_green",

        target_count: 44,
        total_count: 64,
        true_percentage: 100 * 44 / 64,

        image_type: "circles_blue_green",

        target_group: "blue",

        question:
            "Quel pourcentage des cercles étaient bleus ?",

        image:
            "images_matrices/44of64/matrix_circles_blue_green.jpg"
    },

    {
        id: "44_green_blue",

        target_count: 44,
        total_count: 64,
        true_percentage: 100 * 44 / 64,

        image_type: "circles_green_blue",

        target_group: "green",

        question:
            "Quel pourcentage des cercles étaient verts ?",

        image:
            "images_matrices/44of64/matrix_circles_green_blue.jpg"
    },

    {
        id: "44_face",

        target_count: 44,
        total_count: 64,
        true_percentage: 100 * 44 / 64,

        image_type: "face",

        target_group: "black",

        question:
            "Quel pourcentage des personnes étaient noires ?",

        image:
            "images_matrices/44of64/matrix_face.jpg"
    },


    // =====================================================
    // 50 / 64 = 78.125 %
    // =====================================================

    {
        id: "50_blue_green",

        target_count: 50,
        total_count: 64,
        true_percentage: 100 * 50 / 64,

        image_type: "circles_blue_green",

        target_group: "blue",

        question:
            "Quel pourcentage des cercles étaient bleus ?",

        image:
            "images_matrices/50of64/matrix_circles_blue_green.jpg"
    },

    {
        id: "50_green_blue",

        target_count: 50,
        total_count: 64,
        true_percentage: 100 * 50 / 64,

        image_type: "circles_green_blue",

        target_group: "green",

        question:
            "Quel pourcentage des cercles étaient verts ?",

        image:
            "images_matrices/50of64/matrix_circles_green_blue.jpg"
    },

    {
        id: "50_face",

        target_count: 50,
        total_count: 64,
        true_percentage: 100 * 50 / 64,

        image_type: "face",

        target_group: "black",

        question:
            "Quel pourcentage des personnes étaient noires ?",

        image:
            "images_matrices/50of64/matrix_face.jpg"
    },


    // =====================================================
    // 56 / 64 = 87.5 %
    // =====================================================

    {
        id: "56_blue_green",

        target_count: 56,
        total_count: 64,
        true_percentage: 100 * 56 / 64,

        image_type: "circles_blue_green",

        target_group: "blue",

        question:
            "Quel pourcentage des cercles étaient bleus ?",

        image:
            "images_matrices/56of64/matrix_circles_blue_green.jpg"
    },

    {
        id: "56_green_blue",

        target_count: 56,
        total_count: 64,
        true_percentage: 100 * 56 / 64,

        image_type: "circles_green_blue",

        target_group: "green",

        question:
            "Quel pourcentage des cercles étaient verts ?",

        image:
            "images_matrices/56of64/matrix_circles_green_blue.jpg"
    },

    {
        id: "56_face",

        target_count: 56,
        total_count: 64,
        true_percentage: 100 * 56 / 64,

        image_type: "face",

        target_group: "black",

        question:
            "Quel pourcentage des personnes étaient noires ?",

        image:
            "images_matrices/56of64/matrix_face.jpg"
    }

];
// =========================================================
// THREAT QUESTIONNAIRE
// =========================================================

const threatSections = [

    {
        id: "general_immigration_threat",

        title:
            "General immigration threat",

        items: [

            {
                id: "general_threat_1",

                text:
                    "Les personnes noires vivant ici menacent notre mode de vie et nos valeurs en France."
            },

            {
                id: "general_threat_2",

                text:
                    "Les personnes noires vivant ici menacent la sécurité publique en France."
            }

        ]
    },


    {
        id: "collective_existential_threat",

        title:
            "Collective existential threat",

        items: [

            {
                id: "existential_threat_1",

                text:
                    "L'existence de mon groupe racial est en péril."
            },

            {
                id: "existential_threat_2",

                text:
                    "L'existence physique de mon groupe racial est en danger."
            }

        ]
    },


    {
        id: "ingroup_prejudice_concern",

        title:
            "In-group prejudice concern",

        items: [

            {
                id: "prejudice_concern_1",

                text:
                    "La présence des personnes noires entraîne du mécontentement au sein de la population française."
            },

            {
                id: "prejudice_concern_2",

                text:
                    "La présence des personnes noires entraîne une augmentation de la xénophobie en France."
            }

        ]
    },


    {
        id: "fear_of_appearing_racist",

        title:
            "Fear of appearing racist / intergroup anxiety",

        items: [

            {
                id: "intergroup_anxiety_1",

                text:
                    "Quand j'interagis avec une personne noire, il ou elle penserait que j'ai des préjugés envers elle peu importe ce que je fais."
            },

            {
                id: "intergroup_anxiety_2",

                text:
                    "Quand j'interagis avec une personne noire, j'imagine qu'il ou elle observerait attentivement mon comportement pour voir si j'ai des préjugés."
            }

        ]
    }

];


// =========================================================
// STATE
// =========================================================

let participantId = "";

let studyStartedAt = null;


// Images

let currentImageTrial = 0;

let imageRatings = [];

let imagePresentationStartedAt = null;

let imageSliderStartedAt = null;

let imageSliderWasMoved = false;


// Population estimate

let populationEstimate = null;

let populationResponseTimeMs = null;

let populationSliderStartedAt = null;

let populationSliderWasMoved = false;


// Group apart

let groupApartResponse = null;

let groupApartStartedAt = null;


// Threat questionnaire

let currentThreatSection = 0;

let threatResponses = [];

let threatSectionStartedAt = null;


// JATOS

let jatosAvailable = false;


// =========================================================
// JATOS INITIALIZATION
// =========================================================

if (typeof jatos !== "undefined") {

    jatos.onLoad(function () {

        console.log("JATOS chargé");

        jatosAvailable = true;

        initializeParticipantId();

    });

} else {

    console.warn("JATOS non détecté — mode local.");

    initializeParticipantId();


}


// =========================================================
// HELPERS
// =========================================================

function hideAllSections() {

    const sections = document.querySelectorAll(
        ".study-section"
    );

    sections.forEach(section => {

        section.classList.add("hidden");

    });

}


function showOnly(sectionId) {

    hideAllSections();

    document
        .getElementById(sectionId)
        .classList.remove("hidden");

}


function createLikertScale(name) {

    let html = "";


    for (
        let value = 1;
        value <= 7;
        value++
    ) {

        html += `

            <label class="likert-option">

                <input
                    type="radio"
                    name="${name}"
                    value="${value}"
                >

                <span class="likert-number">
                    ${value}
                </span>

            </label>

        `;
    }


    return html;
}

function initializeParticipantId() {

    let urlParticipantId = null;


    // Cas JATOS
    if (
        typeof jatos !== "undefined" &&
        jatosAvailable &&
        jatos.urlQueryParameters
    ) {

        urlParticipantId =
            jatos.urlQueryParameters.participant_id ||
            jatos.urlQueryParameters.participant ||
            jatos.urlQueryParameters.pid ||
            jatos.urlQueryParameters.id ||
            null;
    }


    // Fallback mode local / navigateur
    if (!urlParticipantId) {

        const params =
            new URLSearchParams(
                window.location.search
            );

        urlParticipantId =
            params.get("participant_id") ||
            params.get("participant") ||
            params.get("pid") ||
            params.get("id");
    }


    if (urlParticipantId) {

        participantId =
            String(urlParticipantId).trim();


        const input =
            document.getElementById(
                "participant-id"
            );


        input.value =
            participantId;

        input.disabled =
            true;


        console.log(
            "Participant ID récupéré depuis l'URL :",
            participantId
        );

    }

}
function shuffleArray(array) {

    for (let i = array.length - 1; i > 0; i--) {

        const j =
            Math.floor(Math.random() * (i + 1));

        [array[i], array[j]] =
            [array[j], array[i]];
    }

    return array;
}
// =========================================================
// START
// =========================================================

function startStudy() {

    const input =
        document.getElementById(
            "participant-id"
        );


    if (!participantId) {

            participantId =
            input.value.trim();

    }


    if (!participantId) {

        alert(
            "Veuillez entrer votre identifiant participant."
        );

        return;
    }


    studyStartedAt =
        new Date();


    startImageInstructions();
}


// =========================================================
// IMAGE INSTRUCTIONS
// =========================================================

function startImageInstructions() {

    showOnly(
        "image-instruction-section"
    );


    setTimeout(
        function () {

            startImageTask();

        },

        IMAGE_INSTRUCTION_DURATION_MS
    );

}


// =========================================================
// IMAGE TASK
// =========================================================

function startImageTask() {

    currentImageTrial = 0;

    showOnly(
        "image-rating-section"
    );

    showImageTrial();
}


// =========================================================
// SHOW IMAGE TRIAL
// =========================================================

function showImageTrial() {

    const trial =
        imageTrials[currentImageTrial];


    document
        .getElementById(
            "image-progress"
        )
        .textContent =
        `Image ${currentImageTrial + 1} sur ${imageTrials.length}`;


    const image =
        document.getElementById(
            "stimulus-image"
        );


    const stimulusContainer =
        document.getElementById(
            "stimulus-container"
        );


    const sliderContainer =
        document.getElementById(
            "image-slider-container"
        );


    const slider =
        document.getElementById(
            "image-percentage-slider"
        );


    const sliderValue =
        document.getElementById(
            "image-slider-value"
        );


    const nextButton =
        document.getElementById(
            "image-next-button"
        );


    const loadingMessage =
        document.getElementById(
            "image-loading-message"
        );


    // Reset
    const randomStart =
        Math.floor(Math.random() * 101);

    slider.value = randomStart;

    trial.slider_start =
    randomStart;

    sliderValue.textContent = "— %";

    imageSliderWasMoved =
        false;

    nextButton.disabled =
        true;


    sliderContainer
        .classList
        .add("hidden");


    image.classList.add(
        "hidden"
    );


    loadingMessage
        .classList
        .remove("hidden");


    stimulusContainer
        .classList
        .remove("hidden");


    // Image correctly loaded

    image.onload = function () {

        console.log(
            "Image chargée :",
            trial.image
        );


        loadingMessage
            .classList
            .add("hidden");


        image
            .classList
            .remove("hidden");


        imagePresentationStartedAt =
            performance.now();


        // 2 seconds AFTER actual loading

        setTimeout(
            function () {

                image
                    .classList
                    .add("hidden");


                stimulusContainer
                    .classList
                    .add("hidden");


                sliderContainer
                    .classList
                    .remove("hidden");


                imageSliderStartedAt =
                    performance.now();

            },

            IMAGE_PRESENTATION_DURATION_MS
        );

    };


    image.onerror = function () {

        loadingMessage.textContent =
            "Erreur : impossible de charger cette image.";

        console.error(
            "Image introuvable :",
            trial.image
        );

    };


    image.src =
        trial.image;
}


// =========================================================
// IMAGE SLIDER EVENT
// =========================================================

const imageSlider =
    document.getElementById(
        "image-percentage-slider"
    );


imageSlider.addEventListener(
    "input",
    function () {

        imageSliderWasMoved =
            true;


        document
            .getElementById(
                "image-slider-value"
            )
            .textContent =
            `${imageSlider.value} %`;


        document
            .getElementById(
                "image-next-button"
            )
            .disabled =
            false;

    }
);


// =========================================================
// SUBMIT IMAGE
// =========================================================

async function submitImageRating() {

    if (!imageSliderWasMoved) {

        return;
    }


    const trial =
        imageTrials[currentImageTrial];


    const slider =
        document.getElementById(
            "image-percentage-slider"
        );


    const responseTimeMs =
        performance.now() -
        imageSliderStartedAt;


    imageRatings.push({

        trial_id:
            trial.id,

        condition:
            trial.condition,

        image_type:
            trial.image_type,

        image:
            trial.image,

        percentage:
            Number(slider.value),

        image_duration_ms:
            IMAGE_PRESENTATION_DURATION_MS,

        response_time_ms:
            Math.round(responseTimeMs),

        timestamp_utc:
            new Date().toISOString(),

        slider_start:
            trial.slider_start

    });


    await saveResults(false);


    currentImageTrial++;


    if (
        currentImageTrial <
        imageTrials.length
    ) {

        showImageTrial();

    } else {

        startPopulationQuestion();

    }

}


// =========================================================
// POPULATION ESTIMATE
// =========================================================

function startPopulationQuestion() {

    showOnly(
        "population-section"
    );


    populationSliderStartedAt =
        performance.now();


    populationSliderWasMoved =
        false;


    const slider =
        document.getElementById(
            "population-slider"
        );


    slider.value =
        50;


    document
        .getElementById(
            "population-slider-value"
        )
        .textContent =
        "— %";


    document
        .getElementById(
            "population-next-button"
        )
        .disabled =
        true;
}


// =========================================================
// POPULATION SLIDER
// =========================================================

const populationSlider =
    document.getElementById(
        "population-slider"
    );


populationSlider.addEventListener(
    "input",
    function () {

        populationSliderWasMoved =
            true;


        document
            .getElementById(
                "population-slider-value"
            )
            .textContent =
            `${populationSlider.value} %`;


        document
            .getElementById(
                "population-next-button"
            )
            .disabled =
            false;

    }
);


// =========================================================
// SUBMIT POPULATION ESTIMATE
// =========================================================

async function submitPopulationEstimate() {

    if (!populationSliderWasMoved) {

        return;
    }


    populationEstimate =
        Number(
            document
                .getElementById(
                    "population-slider"
                )
                .value
        );


    populationResponseTimeMs =
        Math.round(
            performance.now() -
            populationSliderStartedAt
        );


    await saveResults(false);


    startGroupApartQuestion();
}


// =========================================================
// GROUP APART
// =========================================================

function startGroupApartQuestion() {

    showOnly(
        "group-apart-section"
    );


    document
        .getElementById(
            "group-apart-scale"
        )
        .innerHTML =
        createLikertScale(
            "group_apart"
        );


    groupApartStartedAt =
        performance.now();

}


// =========================================================
// SUBMIT GROUP APART
// =========================================================

async function submitGroupApart() {

    const selected =
        document.querySelector(
            'input[name="group_apart"]:checked'
        );


    if (!selected) {

        alert(
            "Veuillez sélectionner une réponse."
        );

        return;
    }


    groupApartResponse = {

        item_id:
            "group_apart",

        response:
            Number(selected.value),

        response_time_ms:
            Math.round(
                performance.now() -
                groupApartStartedAt
            ),

        timestamp_utc:
            new Date().toISOString()

    };


    await saveResults(false);


    currentThreatSection =
        0;


    startThreatQuestionnaire();
}


// =========================================================
// THREAT QUESTIONNAIRE
// =========================================================

function startThreatQuestionnaire() {

    showOnly(
        "threat-section"
    );


    showThreatSection();
}


// =========================================================
// SHOW THREAT SECTION
// =========================================================

function showThreatSection() {

    const section =
        threatSections[
            currentThreatSection
        ];


    document
        .getElementById(
            "threat-progress"
        )
        .textContent =
        `Rubrique ${currentThreatSection + 1} sur ${threatSections.length}`;


    document
        .getElementById(
            "threat-title"
        )
        .textContent =
        section.title;


    const container =
        document.getElementById(
            "threat-items"
        );


    container.innerHTML =
        "";


    section.items.forEach(
        (item, index) => {

            const block =
                document.createElement(
                    "div"
                );


            block.className =
                "item-block";


            const radioName =
                `${section.id}_${item.id}`;


            block.innerHTML = `

                <div class="item-number">

                    Affirmation ${index + 1}

                </div>


                <p class="item-text">

                    ${item.text}

                </p>


                <div class="likert-scale">

                    ${createLikertScale(
                        radioName
                    )}

                </div>


                <div class="anchors">

                    <span>
                        1 — Pas du tout d'accord
                    </span>

                    <span>
                        7 — Totalement d'accord
                    </span>

                </div>

            `;


            container.appendChild(
                block
            );

        }
    );


    const button =
        document.getElementById(
            "threat-next-button"
        );


    button.textContent =
        currentThreatSection ===
        threatSections.length - 1
            ? "Terminer"
            : "Suivant";


    threatSectionStartedAt =
        performance.now();


    window.scrollTo(
        0,
        0
    );

}


// =========================================================
// SUBMIT THREAT SECTION
// =========================================================

async function submitThreatSection() {

    const section =
        threatSections[
            currentThreatSection
        ];


    const sectionResponses =
        [];


    for (
        const item
        of section.items
    ) {

        const radioName =
            `${section.id}_${item.id}`;


        const selected =
            document.querySelector(
                `input[name="${radioName}"]:checked`
            );


        if (!selected) {

            alert(
                "Veuillez répondre à toutes les affirmations avant de continuer."
            );

            return;
        }


        sectionResponses.push({

            construct:
                section.id,

            item_id:
                item.id,

            response:
                Number(
                    selected.value
                ),

            timestamp_utc:
                new Date().toISOString()

        });

    }


    const elapsed =
        Math.round(
            performance.now() -
            threatSectionStartedAt
        );


    sectionResponses.forEach(
        response => {

            response.section_response_time_ms =
                elapsed;

        }
    );


    threatResponses.push(
        ...sectionResponses
    );


    await saveResults(false);


    currentThreatSection++;


    if (
        currentThreatSection <
        threatSections.length
    ) {

        showThreatSection();

    } else {

        finishStudy();

    }

}


// =========================================================
// BUILD RESULT JSON
// =========================================================

function buildResultData(completed) {

    return {

        participant_id:
            participantId,


        completed:
            completed,


        study_started_at_utc:
            studyStartedAt
                ? studyStartedAt.toISOString()
                : null,


        completed_at_utc:
            completed
                ? new Date().toISOString()
                : null,


        image_ratings:
            imageRatings,


        population_black_estimate: {

            percentage:
                populationEstimate,

            response_time_ms:
                populationResponseTimeMs

        },


        group_apart:
            groupApartResponse,


        threat_responses:
            threatResponses

    };

}


// =========================================================
// SAVE
// =========================================================

async function saveResults(completed) {

    const data =
        buildResultData(
            completed
        );


    console.log(
        "Données :",
        data
    );


    if (jatosAvailable) {

        await jatos.submitResultData(
            data
        );

        console.log(
            "Sauvegardé dans JATOS."
        );

    } else {

        localStorage.setItem(

            "questionnaire_result",

            JSON.stringify(
                data
            )

        );

        console.log(
            "Sauvegardé localement."
        );

    }

}


// =========================================================
// FINISH
// =========================================================

async function finishStudy() {

    showOnly(
        "saving-section"
    );


    try {

        await saveResults(
            true
        );


        showOnly(
            "end-section"
        );

    } catch (error) {

        console.error(
            "Erreur de sauvegarde finale :",
            error
        );


        alert(
            "Une erreur est survenue lors de l'enregistrement. Veuillez réessayer."
        );


        showOnly(
            "threat-section"
        );

    }

}