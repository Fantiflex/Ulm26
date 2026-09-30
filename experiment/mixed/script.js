console.log("script.js chargé");


// =========================================================
// EXPERIMENT PARAMETERS
// =========================================================



const IMAGE_PRESENTATION_DURATION_MS = 2000;
const IMAGE_RESPONSE_LIMIT_MS = 10000;
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


    // Les trials seront construits au début de l'expérience.
    let imageTrials = [];


    // Il n'y a plus de manipulation entre-sujets sur la couleur
    // des cercles : tous les participants voient le même fichier,
    // "circles.png", où les cercles sont en niveaux de gris
    // (visages noirs -> cercles FONCÉS, visages blancs -> cercles CLAIRS).
    // On garde ce nom de variable pour ne pas casser le reste du code
    // (identifiants de trial, champ "color_scheme" enregistré dans
    // les données), mais sa valeur est fixe.
    const participantCircleScheme = "grey";


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
                        `../images_matrices/${folder}/${version}`;


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


                    // Le mot-clé (noirs/blancs) est mis en gras
                    // pour bien ressortir dans la consigne affichée.
                    // NB : cette chaîne contient du HTML (balise
                    // <strong>) et doit être injectée via innerHTML,
                    // jamais via textContent (voir showImageTrial()).
                    const faceQuestion =
                        askPrimaryGroup
                            ? "Quel pourcentage des visages étaient perçus comme <strong>noirs</strong> ?"
                            : "Quel pourcentage des visages étaient perçus comme <strong>blancs</strong> ?";


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
                            `${basePath}/face.png`

                    });


                    // =================================================
                    // 2. MATRICE NON-SOCIALE
                    // =================================================

                    // Un seul fichier de cercles par matrice
                    // (généré par 03_generate_matrices.py).
                    const circleFile =
                        "circles.png";


                    let circleTargetGroup;
                    let circleQuestion;
                    let circleAskedGroupCount;


                    if (askPrimaryGroup) {

                        // targetCount = nombre de visages noirs
                        // = nombre de cercles FONCÉS dans circles.png
                        circleTargetGroup =
                            "dark";

                        circleQuestion =
                            "Quel pourcentage des cercles étaient perçus comme <strong>foncés</strong> ?";

                        circleAskedGroupCount =
                            targetCount;

                    } else {

                        // Le reste = visages blancs = cercles CLAIRS
                        circleTargetGroup =
                            "light";

                        circleQuestion =
                            "Quel pourcentage des cercles étaient perçus comme <strong>clairs</strong> ?";

                        circleAskedGroupCount =
                            MATRIX_SIZE - targetCount;

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
// DEBRIEF EN ENTONNOIR (juste après la tâche, 1 question par page)
// =========================================================
// Du plus ouvert au plus explicite. Pas de retour en arrière :
// le participant ne voit jamais la question suivante avant
// d'avoir répondu à la précédente.
//   type "text"       : réponse libre
//   type "yesno_text" : Oui / Non + précision libre si Oui
//   type "scale"      : échelle à choix unique

const debriefQuestions = [

    {
        id: "debrief_1_study_goal",
        type: "text",
        text: "Selon vous, qu'est-ce que cette étude cherchait à mesurer ?"
    },

    {
        id: "debrief_2_strategy",
        type: "text",
        text: "Comment avez-vous procédé pour estimer les pourcentages ? Avez-vous utilisé une stratégie particulière ?"
    },

    {
        id: "debrief_3_circles_noticed",
        type: "text",
        text: "Avez-vous remarqué quelque chose de particulier à propos des grilles de cercles ?"
    },

    {
        id: "debrief_4_circles_association",
        type: "yesno_text",
        text: "En regardant les grilles de cercles, les cercles vous ont-ils fait penser à quelque chose ?",
        followup: "Si oui, à quoi ?"
    },

    {
        id: "debrief_5_faces_circles_link",
        type: "yesno_text",
        text: "Avez-vous fait un lien entre les grilles de visages et les grilles de cercles ?",
        followup: "Si oui, lequel ?"
    },

    {
        id: "debrief_6_explicit_projection",
        type: "scale",
        text: "Pendant la tâche, vous est-il arrivé de voir les cercles <strong>foncés</strong> comme représentant des <strong>personnes noires</strong>, et les cercles <strong>clairs</strong> comme représentant des <strong>personnes blanches</strong> ?",
        options: [
            "Jamais",
            "Rarement",
            "Parfois",
            "Souvent",
            "Tout le temps"
        ]
    },


    // --- Contrôle technique : affichage des images ---
    {
        id: "check_7_image_display",
        type: "scale",
        vertical: true,
        text: "Les images se sont-elles bien affichées pendant la tâche (sans lenteur, saccade ou image manquante) ?",
        options: [
            "Oui, toujours",
            "Oui, la plupart du temps",
            "Il y a eu quelques problèmes",
            "Il y a eu beaucoup de problèmes"
        ],
        details: "Si vous avez rencontré des problèmes, pouvez-vous les décrire ? (facultatif)"
    },


    // --- Commentaire libre ---
    {
        id: "comment_8_free",
        type: "text",
        optional: true,
        text: "Avez-vous des remarques ou des commentaires sur l'expérience ? (facultatif)"
    }

];


// =========================================================
// STATE
// =========================================================
let imageResponseTimeoutId = null;
let imageTrialResolved = false;
// Fin de l'étude : retour vers Prolific (code de complétion de l'étude)
const PROLIFIC_COMPLETION_CODE = "C1Q6XPPW";

const PROLIFIC_COMPLETION_URL =
    `https://app.prolific.com/submissions/complete?cc=${PROLIFIC_COMPLETION_CODE}`;

const REDIRECT_DELAY_MS = 3000;


// Identifiants Prolific (voir initializeParticipantId)
const PROLIFIC_ID_LENGTH = 24;

let prolificPidUrl = null;        // lu dans l'URL (PROLIFIC_PID)

let prolificStudyIdUrl = null;    // lu dans l'URL (STUDY_ID)

let prolificSessionIdUrl = null;  // lu dans l'URL (SESSION_ID)

let prolificIdTyped = null;       // saisi par le participant

let studyStartedAt = null;


// Écran / fenêtre du participant (mesuré au début et à la fin)
let screenInfoStart = null;

let screenInfoEnd = null;


// Images

let currentImageTrial = 0;

let imageRatings = [];

// Training

// 2 essais d'entraînement : 1 social (visages) + 1 non-social (cercles)
const N_TRAINING_TRIALS = 2;

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


// Debrief en entonnoir

let currentDebriefQuestion = 0;

let debriefResponses = [];

let debriefQuestionStartedAt = null;


// Demographics (dernière page : genre, âge, niveau d'études)

const AGE_MIN = 18;

const AGE_MAX = 80;

let demographicsResponse = null;

let demographicsStartedAt = null;


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


// ---------------------------------------------------------
// IDENTIFIANTS PROLIFIC
// ---------------------------------------------------------
// Deux sources, enregistrées sous des noms DIFFÉRENTS pour
// pouvoir les comparer après coup :
//   - prolific_pid_url   : lu automatiquement dans l'URL du lien
//                          Prolific (paramètres PROLIFIC_PID,
//                          STUDY_ID, SESSION_ID) ;
//   - prolific_id_typed  : saisi par le participant sur la
//                          première page (vérification / secours).
// La case de saisie n'est JAMAIS pré-remplie avec l'ID de l'URL,
// sinon la vérification croisée n'aurait aucun sens.

function readUrlParameter(name) {

    // Cas JATOS : JATOS conserve les paramètres du lien d'origine
    if (
        typeof jatos !== "undefined" &&
        jatosAvailable &&
        jatos.urlQueryParameters &&
        jatos.urlQueryParameters[name]
    ) {
        return String(jatos.urlQueryParameters[name]).trim();
    }

    // Mode local / navigateur
    const value =
        new URLSearchParams(window.location.search).get(name);

    return value ? String(value).trim() : null;
}


function initializeParticipantId() {

    prolificPidUrl =
        readUrlParameter("PROLIFIC_PID");

    prolificStudyIdUrl =
        readUrlParameter("STUDY_ID");

    prolificSessionIdUrl =
        readUrlParameter("SESSION_ID");

    console.log(
        "IDs Prolific lus dans l'URL :",
        {
            PROLIFIC_PID: prolificPidUrl,
            STUDY_ID: prolificStudyIdUrl,
            SESSION_ID: prolificSessionIdUrl
        }
    );
}


// ---------------------------------------------------------
// TAILLE D'ÉCRAN / FENÊTRE
// ---------------------------------------------------------
// Le navigateur donne des tailles en pixels, PAS en centimètres :
// la taille physique de l'écran (en pouces/cm) n'est pas accessible.
//   screen_*      : taille de l'écran entier
//   viewport_*    : zone réellement visible dans le navigateur
//   device_pixel_ratio : nombre de pixels physiques par pixel CSS
//                        (écrans Retina / zoom du navigateur)

function getScreenInfo() {

    const dpr =
        window.devicePixelRatio || 1;

    return {

        screen_width_px:
            window.screen.width,

        screen_height_px:
            window.screen.height,

        screen_available_width_px:
            window.screen.availWidth,

        screen_available_height_px:
            window.screen.availHeight,

        viewport_width_px:
            window.innerWidth,

        viewport_height_px:
            window.innerHeight,

        device_pixel_ratio:
            dpr,

        screen_width_physical_px:
            Math.round(window.screen.width * dpr),

        screen_height_physical_px:
            Math.round(window.screen.height * dpr),

        orientation:
            (window.screen.orientation && window.screen.orientation.type) || null,

        max_touch_points:
            navigator.maxTouchPoints || 0,

        user_agent:
            navigator.userAgent,

        timestamp_utc:
            new Date().toISOString()

    };
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

    const error =
        document.getElementById(
            "participant-id-error"
        );


    // On enlève les espaces éventuels (copier-coller)
    const typedId =
        input.value.replace(/\s+/g, "");


    // Un ID Prolific fait exactement 24 caractères (lettres/chiffres)
    if (!/^[A-Za-z0-9]{24}$/.test(typedId)) {

        error.textContent =
            typedId.length === 0
                ? "Veuillez indiquer votre ID Prolific."
                : `Votre ID Prolific doit contenir exactement ${PROLIFIC_ID_LENGTH} caractères (lettres et chiffres uniquement). Vous en avez saisi ${typedId.length}.`;

        error.classList.remove("hidden");

        input.focus();

        return;
    }


    error.classList.add("hidden");

    prolificIdTyped =
        typedId;


    // ==============================================
    // Construit les 90 trials
    // ==============================================
    // (plus de condition non-sociale à assigner : tous
    // les participants voient les mêmes cercles, cf.
    // participantCircleScheme = "blue_green" plus haut)

    imageTrials =
        buildImageTrials();


    studyStartedAt =
        new Date();

    screenInfoStart =
        getScreenInfo();


    // Page d'accueil générale (visages / cercles),
    // avant les instructions détaillées de la tâche.
    showOnly(
        "study-intro-section"
    );
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


    // 1 grille de visages + 1 grille de cercles, tirées au hasard
    // parmi les stimuli, présentées dans un ordre aléatoire.
    // On travaille sur des copies afin de ne pas modifier imageTrials.

    const randomFaceTrial =
        shuffleArray(
            imageTrials.filter(t => t.image_type === "face")
        )[0];

    const randomCircleTrial =
        shuffleArray(
            imageTrials.filter(t => t.image_type === "circles")
        )[0];

    trainingTrials =
        shuffleArray(
            [randomFaceTrial, randomCircleTrial]
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

    imageTrialResolved = false;

    if (imageResponseTimeoutId !== null) {
        clearTimeout(imageResponseTimeoutId);
        imageResponseTimeoutId = null;
    }

    // La progression ("Image X sur Y") n'est volontairement
    // plus affichée aux participants (cf. #image-progress,
    // laissé vide dans le HTML).

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

        // innerHTML (et non textContent) : la question contient
        // désormais une balise <strong> pour mettre le mot-clé
        // (noirs/blancs/clairs/foncés) en gras.
        responseInstruction.innerHTML =
            trial.question;

    }

    // -----------------------------------------------------
    // Curseur : aucune position n'est mise en avant visuellement.
    // Un <input type="range"> a techniquement toujours une valeur
    // sous-jacente (ici fixée à 50, le milieu, plutôt que random),
    // mais le curseur (le "thumb") est masqué via la classe CSS
    // "slider-not-touched" tant que le participant n'a pas cliqué
    // ou glissé dessus — dès la première interaction, la classe
    // est retirée et le curseur apparaît là où il a cliqué.
    // Voir style.css pour la règle correspondante.
    // -----------------------------------------------------

    const neutralStart = 50;

    slider.value = neutralStart;

    trial.slider_start =
    neutralStart;

    slider.classList.add(
        "slider-not-touched"
    );

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

        // Temps de chargement de l'image (lag réseau) et taille
        // réellement affichée à l'écran (en pixels CSS).
        trial.image_load_ms =
            Math.round(
                imagePresentationStartedAt -
                trial.image_load_started_at
            );

        const displayedBox =
            image.getBoundingClientRect();

        trial.image_displayed_width_px =
            Math.round(displayedBox.width);

        trial.image_displayed_height_px =
            Math.round(displayedBox.height);


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
            // LIMITE DE 10 SECONDES POUR RÉPONDRE
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


    trial.image_load_started_at =
        performance.now();

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


        // Dès la première interaction, on révèle le curseur
        // (il était masqué par "slider-not-touched").
        imageSlider.classList.remove(
            "slider-not-touched"
        );


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

        clearTimeout(
            imageResponseTimeoutId
        );

        imageResponseTimeoutId =
            null;
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

        asked_group_count:
            trial.asked_group_count,

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

        image_load_ms:
            trial.image_load_ms,

        image_displayed_width_px:
            trial.image_displayed_width_px,

        image_displayed_height_px:
            trial.image_displayed_height_px,

        presentation_order:
            trialIndex + 1,

        timed_out:
            false,

        phase:
            isTraining
                ? "training"
                : "experiment"
    };


    // =============================================
    // STOCKAGE DE LA RÉPONSE
    // =============================================

    if (isTraining) {

        trainingRatings.push(
            rating
        );

    } else {

        imageRatings.push(
            rating
        );

    }


    await saveResults(false);


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

            showQuestionnaireTransition();

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

        asked_group_count:
            trial.asked_group_count,

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

        image_load_ms:
            trial.image_load_ms,

        image_displayed_width_px:
            trial.image_displayed_width_px,

        image_displayed_height_px:
            trial.image_displayed_height_px,

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

                    showQuestionnaireTransition();

                }

            }

        },

        TIMEOUT_WARNING_DURATION_MS
    );

}
// =========================================================
// TRANSITION VERS LES QUESTIONNAIRES
// =========================================================

function showQuestionnaireTransition() {

    showOnly(
        "questionnaire-transition-section"
    );

}


// =========================================================
// DEBRIEF EN ENTONNOIR — une question par page
// =========================================================

function startDebrief() {

    currentDebriefQuestion = 0;

    debriefResponses = [];

    showOnly(
        "debrief-section"
    );

    renderDebriefQuestion();
}


// Supprime les balises HTML (<strong>) pour enregistrer
// le texte de la question en clair dans les données.
function stripHtml(html) {

    const div =
        document.createElement("div");

    div.innerHTML =
        html;

    return div.textContent.trim();
}


function renderDebriefQuestion() {

    const question =
        debriefQuestions[currentDebriefQuestion];

    const container =
        document.getElementById(
            "debrief-question-container"
        );

    let answerHtml = "";


    if (question.type === "text") {

        answerHtml = `
            <textarea
                id="debrief-text"
                class="debrief-textarea"
                rows="5"
                placeholder="Votre réponse"
            ></textarea>
        `;

    } else if (question.type === "yesno_text") {

        answerHtml = `
            <div class="choice-row">

                <label class="choice-option">
                    <input type="radio" name="debrief_yesno" value="oui">
                    <span>Oui</span>
                </label>

                <label class="choice-option">
                    <input type="radio" name="debrief_yesno" value="non">
                    <span>Non</span>
                </label>

            </div>

            <div
                id="debrief-followup-block"
                class="debrief-followup hidden"
            >

                <p class="item-text">
                    ${question.followup}
                </p>

                <textarea
                    id="debrief-text"
                    class="debrief-textarea"
                    rows="4"
                    placeholder="Votre réponse"
                ></textarea>

            </div>
        `;

    } else if (question.type === "scale") {

        answerHtml = `
            <div class="${question.vertical ? "choice-list" : "choice-row debrief-scale"}">
                ${question.options.map((label, index) => `
                    <label class="choice-option">
                        <input type="radio" name="debrief_scale" value="${index + 1}">
                        <span>${label}</span>
                    </label>
                `).join("")}
            </div>
        `;

        // Précision libre facultative sous l'échelle
        if (question.details) {

            answerHtml += `
                <div class="debrief-followup">

                    <p class="item-text">
                        ${question.details}
                    </p>

                    <textarea
                        id="debrief-text"
                        class="debrief-textarea"
                        rows="3"
                        placeholder="Votre réponse (facultatif)"
                    ></textarea>

                </div>
            `;
        }
    }


    container.innerHTML = `
        <div class="question-card">

            <h2>
                ${question.text}
            </h2>

            ${answerHtml}

        </div>
    `;


    // Oui -> affiche la précision ; Non -> la masque
    if (question.type === "yesno_text") {

        document
            .querySelectorAll('input[name="debrief_yesno"]')
            .forEach(input => {

                input.addEventListener(
                    "change",
                    function () {

                        document
                            .getElementById("debrief-followup-block")
                            .classList.toggle(
                                "hidden",
                                this.value !== "oui"
                            );

                    }
                );

            });
    }


    debriefQuestionStartedAt =
        performance.now();

    window.scrollTo(0, 0);
}


async function submitDebriefQuestion() {

    const question =
        debriefQuestions[currentDebriefQuestion];

    const response = {

        item_id:
            question.id,

        question_order:
            currentDebriefQuestion + 1,

        question_text:
            stripHtml(question.text),

        type:
            question.type

    };


    if (question.type === "text") {

        const text =
            document.getElementById("debrief-text").value.trim();

        // Les questions facultatives (commentaire libre) peuvent rester vides
        if (!text && !question.optional) {

            alert("Veuillez répondre à la question avant de continuer.");

            return;
        }

        response.text = text || null;

    } else if (question.type === "yesno_text") {

        const selected =
            document.querySelector('input[name="debrief_yesno"]:checked');

        if (!selected) {

            alert("Veuillez répondre Oui ou Non.");

            return;
        }

        const text =
            document.getElementById("debrief-text").value.trim();

        if (selected.value === "oui" && !text) {

            alert("Veuillez préciser votre réponse.");

            return;
        }

        response.yes_no = selected.value;

        response.text = selected.value === "oui" ? text : null;

    } else if (question.type === "scale") {

        const selected =
            document.querySelector('input[name="debrief_scale"]:checked');

        if (!selected) {

            alert("Veuillez sélectionner une réponse.");

            return;
        }

        response.scale_value = Number(selected.value);

        response.scale_label =
            question.options[Number(selected.value) - 1];

        if (question.details) {

            response.text =
                document.getElementById("debrief-text").value.trim() || null;
        }
    }


    response.response_time_ms =
        Math.round(
            performance.now() -
            debriefQuestionStartedAt
        );

    response.timestamp_utc =
        new Date().toISOString();


    debriefResponses.push(
        response
    );


    await saveResults(false);


    currentDebriefQuestion++;


    if (currentDebriefQuestion < debriefQuestions.length) {

        renderDebriefQuestion();

    } else {

        // Fin du debrief -> pourcentage de personnes noires en France.
        startPopulationQuestion();
    }
}


// =========================================================
// POPULATION ESTIMATE (après le debrief)
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


    // Même logique que pour le curseur des images : valeur
    // technique neutre (50) et thumb masqué via la classe
    // "slider-not-touched" tant que le participant n'a pas
    // cliqué/glissé dessus (voir style.css).
    const neutralStart = 50;

    slider.value =
        neutralStart;

    slider.classList.add(
        "slider-not-touched"
    );


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


        // Dès la première interaction, on révèle le curseur
        // (il était masqué par "slider-not-touched").
        populationSlider.classList.remove(
            "slider-not-touched"
        );


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


    // Après la population -> page démographique (dernière page).
    startDemographics();
}


// =========================================================
// DEMOGRAPHICS — genre, âge, niveau d'études (une seule page)
// =========================================================

function startDemographics() {

    showOnly(
        "demographics-section"
    );


    document
        .getElementById("demo-age-error")
        .classList.add("hidden");


    demographicsStartedAt =
        performance.now();


    window.scrollTo(0, 0);
}


// Renvoie l'âge (entier) s'il est valide, sinon null.
function readValidAge() {

    const raw =
        document
            .getElementById("demo-age")
            .value
            .trim();


    // Uniquement des chiffres (pas de décimales, pas de texte)
    if (!/^\d+$/.test(raw)) {
        return null;
    }


    const age =
        Number(raw);


    if (age < AGE_MIN || age > AGE_MAX) {
        return null;
    }


    return age;
}


async function submitDemographics() {

    const gender =
        document.querySelector(
            'input[name="demo_gender"]:checked'
        );

    const education =
        document.querySelector(
            'input[name="demo_education"]:checked'
        );

    const device =
        document.querySelector(
            'input[name="demo_device"]:checked'
        );

    const age =
        readValidAge();

    const ageError =
        document.getElementById(
            "demo-age-error"
        );


    if (age === null) {

        ageError.classList.remove("hidden");

        document
            .getElementById("demo-age")
            .focus();

    } else {

        ageError.classList.add("hidden");

    }


    if (!gender || !education || !device || age === null) {

        alert(
            age === null
                ? `Veuillez indiquer votre âge en années (nombre entier entre ${AGE_MIN} et ${AGE_MAX}).`
                : "Veuillez répondre à toutes les questions avant de terminer."
        );

        return;
    }


    demographicsResponse = {

        gender:
            gender.value,

        age:
            age,

        education:
            education.value,

        device:
            device.value,

        response_time_ms:
            Math.round(
                performance.now() -
                demographicsStartedAt
            ),

        timestamp_utc:
            new Date().toISOString()

    };


    finishStudy();
}


// =========================================================
// BUILD RESULT JSON
// =========================================================

function buildResultData(completed) {

    return {

        // ID lus automatiquement dans le lien Prolific
        prolific_pid_url:
            prolificPidUrl,

        prolific_study_id_url:
            prolificStudyIdUrl,

        prolific_session_id_url:
            prolificSessionIdUrl,


        // ID saisi par le participant (vérification)
        prolific_id_typed:
            prolificIdTyped,


        // true / false si les deux sont disponibles, sinon null
        prolific_ids_match:
            prolificPidUrl && prolificIdTyped
                ? prolificPidUrl === prolificIdTyped
                : null,


        completed:
            completed,


        screen_info_start:
            screenInfoStart,

        screen_info_end:
            screenInfoEnd,


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


        debrief:
            debriefResponses,


        population_black_estimate: {

            percentage:
                populationEstimate,

            response_time_ms:
                populationResponseTimeMs

        },


        demographics:
            demographicsResponse

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

    screenInfoEnd =
        getScreenInfo();


    try {

        await saveResults(
            true
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
            "demographics-section"
        );

        return;
    }


    // Page de fin : remerciement + lien de secours vers Prolific
    // (au cas où la redirection automatique ne fonctionnerait pas).
    document
        .getElementById("prolific-return-link")
        .href = PROLIFIC_COMPLETION_URL;

    document
        .getElementById("prolific-completion-code")
        .textContent = PROLIFIC_COMPLETION_CODE;

    showOnly(
        "end-section"
    );


    // Redirection automatique vers Prolific après quelques secondes.
    setTimeout(
        redirectToProlific,
        REDIRECT_DELAY_MS
    );

}


// =========================================================
// RETOUR VERS PROLIFIC
// =========================================================
// Sur JATOS, endStudyAndRedirect() clôt l'étude côté serveur
// (statut FINISHED) PUIS envoie le participant sur Prolific,
// qui enregistre alors la participation comme terminée.

function redirectToProlific() {

    if (jatosAvailable) {

        try {

            jatos.endStudyAndRedirect(
                PROLIFIC_COMPLETION_URL
            );

            return;

        } catch (error) {

            console.error(
                "endStudyAndRedirect a échoué, redirection simple :",
                error
            );
        }
    }

    // Mode local (ou secours)
    window.location.href =
        PROLIFIC_COMPLETION_URL;
}
