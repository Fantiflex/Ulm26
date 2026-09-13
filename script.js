console.log("script.js chargé");


// =========================================================
// EXPERIMENT PARAMETERS
// =========================================================



const IMAGE_PRESENTATION_DURATION_MS = 2000;
const IMAGE_RESPONSE_LIMIT_MS = 5000;
const TIMEOUT_WARNING_DURATION_MS = 1000;
const MATRIX_SIZE = 64;

const MATRIX_ROWS = 8;

const MATRIX_COLUMNS = 8;




// =========================================================
// IMAGE TRIALS
// =========================================================

// Nombre exact de personnes noires parmi 64.
// 8/64 puis incréments de 6 jusqu'à 56/64.
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

    // 5 versions différentes de chaque composition.
    const N_VERSIONS = 5;


    // Les trials seront construits au début de l'expérience,
    // une fois la condition non-sociale du participant choisie.
    let imageTrials = [];


    // Un participant reçoit UNE SEULE condition non-sociale
    // pour toute l'expérience :
    // "blue_green" OU "green_blue".
    let participantCircleScheme = null;


    // =========================================================
    // CHOIX DE LA CONDITION NON-SOCIALE
    // =========================================================

    // Petit hash déterministe à partir du participant ID.
    // Cela permet à un même participant d'avoir toujours
    // la même condition s'il recharge l'expérience.

    function hashString(str) {

        let hash = 0;

        for (let i = 0; i < str.length; i++) {

            hash =
                ((hash << 5) - hash) +
                str.charCodeAt(i);

            hash |= 0;
        }

        return Math.abs(hash);
    }


    function chooseParticipantCircleScheme() {

        const hash =
            hashString(
                participantId || "anonymous"
            );

        return hash % 2 === 0
            ? "blue_green"
            : "green_blue";
    }


    // =========================================================
    // CONSTRUCTION DES 90 TRIALS
    // =========================================================

    function buildImageTrials() {

        const trials = [];

        let pairIndex = 0;


        TARGET_COUNTS.forEach(
            targetCount => {

                // Nombre de personnes/cercle du groupe "principal"
                // dans la matrice.
                const basePercentage =
                    100 * targetCount / MATRIX_SIZE;


                const folderPercentage =
                    Math.round(
                        basePercentage
                    );


                const folder =
                    `${folderPercentage}pct_black`;


                const paddedTargetCount =
                    String(
                        targetCount
                    ).padStart(
                        2,
                        "0"
                    );


                for (
                    let versionNumber = 1;
                    versionNumber <= N_VERSIONS;
                    versionNumber++
                ) {

                    const paddedVersion =
                        String(
                            versionNumber
                        ).padStart(
                            2,
                            "0"
                        );


                    const version =
                        `mb${paddedTargetCount}_n64_v${paddedVersion}`;


                    const basePath =
                        `images_matrices/${folder}/${version}`;


                    // =================================================
                    // On alterne le groupe demandé
                    // =================================================

                    const askPrimaryGroup =
                        pairIndex % 2 === 0;


                    // =================================================
                    // 1. MATRICE SOCIALE
                    // =================================================

                    const faceTargetGroup =
                        askPrimaryGroup
                            ? "black"
                            : "white";


                    const faceQuestion =
                        askPrimaryGroup
                            ? "Quel pourcentage des visages étaient perçus comme noirs ?"
                            : "Quel pourcentage des visages étaient perçus comme blancs ?";


                    const faceAskedGroupCount =
                        askPrimaryGroup
                            ? targetCount
                            : MATRIX_SIZE - targetCount;


                    const faceTruePercentage =
                        100 *
                        faceAskedGroupCount /
                        MATRIX_SIZE;


                    trials.push({

                        id:
                            `mb${paddedTargetCount}_face_${faceTargetGroup}_v${paddedVersion}`,

                        target_count:
                            targetCount,

                        asked_group_count:
                            faceAskedGroupCount,

                        total_count:
                            MATRIX_SIZE,

                        true_percentage:
                            faceTruePercentage,

                        folder_percentage:
                            folderPercentage,

                        version:
                            versionNumber,

                        image_type:
                            "face",

                        color_scheme:
                            null,

                        target_group:
                            faceTargetGroup,

                        question:
                            faceQuestion,

                        image:
                            `${basePath}/face.jpg`

                    });


                    // =================================================
                    // 2. MATRICE NON-SOCIALE
                    // =================================================

                    // IMPORTANT :
                    // le fichier affiché dépend UNIQUEMENT
                    // de la condition attribuée au participant.

                    const circleFile =
                        participantCircleScheme === "blue_green"
                            ? "circles_blue_green.jpg"
                            : "circles_green_blue.jpg";


                    let circleTargetGroup;
                    let circleQuestion;
                    let circleAskedGroupCount;


                    // -------------------------------------------------
                    // CONDITION BLUE_GREEN
                    // -------------------------------------------------

                    if (
                        participantCircleScheme === "blue_green"
                    ) {

                        if (askPrimaryGroup) {

                            // targetCount correspond aux bleus
                            circleTargetGroup =
                                "blue";

                            circleQuestion =
                                "Quel pourcentage des cercles étaient bleus ?";

                            circleAskedGroupCount =
                                targetCount;

                        } else {

                            // Le reste correspond aux verts
                            circleTargetGroup =
                                "green";

                            circleQuestion =
                                "Quel pourcentage des cercles étaient verts ?";

                            circleAskedGroupCount =
                                MATRIX_SIZE - targetCount;

                        }

                    }


                    // -------------------------------------------------
                    // CONDITION GREEN_BLUE
                    // -------------------------------------------------

                    else {

                        if (askPrimaryGroup) {

                            // targetCount correspond aux verts
                            circleTargetGroup =
                                "green";

                            circleQuestion =
                                "Quel pourcentage des cercles étaient verts ?";

                            circleAskedGroupCount =
                                targetCount;

                        } else {

                            // Le reste correspond aux bleus
                            circleTargetGroup =
                                "blue";

                            circleQuestion =
                                "Quel pourcentage des cercles étaient bleus ?";

                            circleAskedGroupCount =
                                MATRIX_SIZE - targetCount;

                        }

                    }


                    const circleTruePercentage =
                        100 *
                        circleAskedGroupCount /
                        MATRIX_SIZE;


                    trials.push({

                        id:
                            `mb${paddedTargetCount}_${participantCircleScheme}_${circleTargetGroup}_v${paddedVersion}`,

                        target_count:
                            targetCount,

                        asked_group_count:
                            circleAskedGroupCount,

                        total_count:
                            MATRIX_SIZE,

                        true_percentage:
                            circleTruePercentage,

                        folder_percentage:
                            folderPercentage,

                        version:
                            versionNumber,

                        image_type:
                            "circles",

                        color_scheme:
                            participantCircleScheme,

                        target_group:
                            circleTargetGroup,

                        question:
                            circleQuestion,

                        image:
                            `${basePath}/${circleFile}`

                    });


                    pairIndex++;
                }

            }
        );


        console.log(
            `${trials.length} trials générés.`
        );


        if (
            trials.length !== 90
        ) {

            console.error(
                "ERREUR : il devrait y avoir exactement 90 trials."
            );

        }


        // =====================================================
        // DIAGNOSTIC
        // =====================================================

        const counts = {};


        trials.forEach(
            trial => {

                const key =
                    `${trial.image_type}_${trial.target_group}`;

                counts[key] =
                    (counts[key] || 0) + 1;

            }
        );


        console.log(
            "Condition cercles du participant :",
            participantCircleScheme
        );


        console.log(
            "Répartition des questions :",
            counts
        );


        return trials;
    }
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
let imageResponseTimeoutId = null;
let imageTrialResolved = false;
let participantId = "";

let studyStartedAt = null;


// Images

let currentImageTrial = 0;

let imageRatings = [];

// Training

const N_TRAINING_TRIALS = 3;

let trainingTrials = [];

let currentTrainingTrial = 0;

let trainingRatings = [];

let isTraining = false;

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
            jatos.urlQueryParameters.PROLIFIC_PID ||
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
            params.get("PROLIFIC_PID") ||
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


    // ==============================================
    // Assigne UNE condition non-sociale au participant
    // ==============================================

    participantCircleScheme =
        chooseParticipantCircleScheme();


    console.log(
        "Condition non-sociale :",
        participantCircleScheme
    );


    // ==============================================
    // Construit les 90 trials
    // ==============================================

    imageTrials =
        buildImageTrials();


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

}

function showSecondInstructions() {

    showOnly(
        "image-second-instruction-section"
    );

}
// =========================================================
// TRAINING
// =========================================================

function startTrainingTask() {

    isTraining = true;

    currentTrainingTrial = 0;

    trainingRatings = [];


    // On prend 3 grilles aléatoires parmi les stimuli disponibles.
    // On travaille sur une copie afin de ne pas modifier imageTrials.

    const shuffledCopy =
        shuffleArray(
            [...imageTrials]
        );


    trainingTrials =
        shuffledCopy.slice(
            0,
            N_TRAINING_TRIALS
        );


    console.log(
        "Trials d'entraînement :",
        trainingTrials
    );


    showOnly(
        "image-rating-section"
    );


    showImageTrial();
}

// =========================================================
// IMAGE TASK
// =========================================================

function startImageTask() {

    isTraining = false;

    currentImageTrial = 0;

    shuffleArray(imageTrials);

    showOnly(
        "image-rating-section"
    );

    showImageTrial();
}


// =========================================================
// SHOW IMAGE TRIAL
// =========================================================
function getCurrentTrial() {

    if (isTraining) {

        return trainingTrials[
            currentTrainingTrial
        ];

    }

    return imageTrials[
        currentImageTrial
    ];
}


function getCurrentTrialIndex() {

    return isTraining
        ? currentTrainingTrial
        : currentImageTrial;
}


function getCurrentTrialTotal() {

    return isTraining
        ? trainingTrials.length
        : imageTrials.length;
}


function showImageTrial() {

    const trial =
        getCurrentTrial();

    const trialIndex =
        getCurrentTrialIndex();

    const trialTotal =
        getCurrentTrialTotal();

    imageTrialResolved = false;

    if (imageResponseTimeoutId !== null) {
        clearTimeout(imageResponseTimeoutId);
        imageResponseTimeoutId = null;
    }

    document
        .getElementById(
            "image-progress"
        )
        .textContent =
        isTraining
            ? `Entraînement ${trialIndex + 1} sur ${trialTotal}`
            : `Image ${trialIndex + 1} sur ${trialTotal}`;

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

    const responseInstruction =
    document.querySelector(
        "#image-slider-container .response-instruction"
    );


    if (responseInstruction) {

        responseInstruction.textContent =
            trial.question;

    }

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


            // =====================================================
            // LIMITE DE 5 SECONDES POUR RÉPONDRE
            // =====================================================

            imageResponseTimeoutId =
                setTimeout(
                    handleImageTimeout,
                    IMAGE_RESPONSE_LIMIT_MS
                );

                        },

                    IMAGE_PRESENTATION_DURATION_MS
                );

    };


    image.onerror = function () {

        loadingMessage.textContent =
            "Erreur : impossible de charger cette image.";

        console.error(
            "IMAGE INTROUVABLE :",
            trial.image
        );

        console.error(
            "URL PAGE :",
            window.location.href
        );

        console.error(
            "URL IMAGE RÉSOLUE :",
            new URL(
                trial.image,
                window.location.href
            ).href
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

    if (
        !imageSliderWasMoved ||
        imageTrialResolved
    ) {
        return;
    }

    imageTrialResolved = true;

    if (imageResponseTimeoutId !== null) {
        clearTimeout(imageResponseTimeoutId);
        imageResponseTimeoutId = null;
    }


    const trial =
        getCurrentTrial();

    const trialIndex =
        getCurrentTrialIndex();


    const slider =
        document.getElementById(
            "image-percentage-slider"
        );


    const responseTimeMs =
        performance.now() -
        imageSliderStartedAt;


    const rating = {

        trial_id:
            trial.id,

        target_count:
            trial.target_count,

        total_count:
            trial.total_count,

        true_percentage:
            trial.true_percentage,

        folder_percentage:
            trial.folder_percentage,

        version:
            trial.version,

        image_type:
            trial.image_type,

        color_scheme:
            trial.color_scheme,

        target_group:
            trial.target_group,

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
            trial.slider_start,

        presentation_order:
            trialIndex + 1,

        timed_out:
            false,

        phase:
            isTraining
                ? "training"
                : "experiment"
    };


    if (isTraining) {

        trainingRatings.push(rating);

    } else {

        imageRatings.push(rating);

    }


    await saveResults(false);


    if (
        currentTrainingTrial <
        trainingTrials.length
    ) {

        showImageTrial();

    } else {

        showOnly(
            "training-complete-section"
        );

    }

    } else {

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
}
// =========================================================
// IMAGE RESPONSE TIMEOUT
// =========================================================

async function handleImageTimeout() {

    // Évite qu'un trial déjà validé soit traité une seconde fois.
    if (imageTrialResolved) {
        return;
    }

    imageTrialResolved = true;
    imageResponseTimeoutId = null;


    const trial =
        getCurrentTrial();

    const trialIndex =
        getCurrentTrialIndex();


    // -----------------------------------------------------
    // Construit la réponse timeout
    // -----------------------------------------------------

    const rating = {

        trial_id:
            trial.id,

        target_count:
            trial.target_count,

        total_count:
            trial.total_count,

        true_percentage:
            trial.true_percentage,

        folder_percentage:
            trial.folder_percentage,

        version:
            trial.version,

        image_type:
            trial.image_type,

        color_scheme:
            trial.color_scheme,

        target_group:
            trial.target_group,

        image:
            trial.image,

        // Pas de réponse valide
        percentage:
            null,

        image_duration_ms:
            IMAGE_PRESENTATION_DURATION_MS,

        response_time_ms:
            IMAGE_RESPONSE_LIMIT_MS,

        timed_out:
            true,

        timestamp_utc:
            new Date().toISOString(),

        slider_start:
            trial.slider_start,

        presentation_order:
            trialIndex + 1,

        phase:
            isTraining
                ? "training"
                : "experiment"
    };


    // -----------------------------------------------------
    // Stocke séparément entraînement et vraie expérience
    // -----------------------------------------------------

    if (isTraining) {

        trainingRatings.push(
            rating
        );

    } else {

        imageRatings.push(
            rating
        );

    }


    // -----------------------------------------------------
    // Cache le slider
    // -----------------------------------------------------

    const sliderContainer =
        document.getElementById(
            "image-slider-container"
        );

    sliderContainer
        .classList
        .add("hidden");


    // -----------------------------------------------------
    // Message rouge
    // -----------------------------------------------------

    let warning =
        document.getElementById(
            "timeout-warning"
        );


    if (!warning) {

        warning =
            document.createElement("div");

        warning.id =
            "timeout-warning";

        warning.textContent =
            "Attention, vous avez pris trop longtemps à répondre.";

        document
            .getElementById(
                "image-rating-section"
            )
            .appendChild(
                warning
            );
    }


    warning.classList.remove(
        "hidden"
    );


    // -----------------------------------------------------
    // Sauvegarde
    // -----------------------------------------------------

    await saveResults(false);


    // -----------------------------------------------------
    // Attend brièvement puis passe au trial suivant
    // -----------------------------------------------------

    setTimeout(
        function () {

            warning.classList.add(
                "hidden"
            );


            // =============================================
            // ENTRAÎNEMENT
            // =============================================

            if (isTraining) {

                currentTrainingTrial++;


                if (
                currentTrainingTrial <
                trainingTrials.length
            ) {

                showImageTrial();

            } else {

                showOnly(
                    "training-complete-section"
                );

            }

            }


            // =============================================
            // VRAIE EXPÉRIENCE
            // =============================================

            else {

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

        },

        TIMEOUT_WARNING_DURATION_MS
    );

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


    const randomStart =
        Math.floor(
            Math.random() * 101
        );

    slider.value =
        randomStart;


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

        circle_condition:
        participantCircleScheme,


        expected_training_trials:
            N_TRAINING_TRIALS,

        completed_training_trials:
            trainingRatings.length,

        training_ratings:
            trainingRatings,


        expected_image_trials:
            90,

        completed_image_trials:
            imageRatings.length,

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