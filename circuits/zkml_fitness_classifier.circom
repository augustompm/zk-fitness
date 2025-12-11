/*
zkml_fitness_classifier.circom

ZKP circuit with embedded Decision Tree classifier for exercise intensity.
Trained on MHEALTH multi-sensor data.

References:
    [1] WHO Physical Activity Guidelines 2020
    [12] ACSM Guidelines 10th ed
    [13] MHEALTH Dataset (Banos et al. 2014)
    [14] Groth16 (Groth 2016)
    [21] CART (Breiman et al. 1984)
*/

pragma circom 2.0.0;

template Num2Bits(n) {
    signal input in;
    signal output out[n];

    var lc = 0;
    var e2 = 1;
    for (var i = 0; i < n; i++) {
        out[i] <-- (in >> i) & 1;
        out[i] * (out[i] - 1) === 0;
        lc += out[i] * e2;
        e2 = e2 * 2;
    }
    lc === in;
}

template LessThan(n) {
    signal input in[2];
    signal output out;

    component num2bits = Num2Bits(n + 1);
    num2bits.in <== (1 << n) + in[0] - in[1];
    out <== 1 - num2bits.out[n];
}

template LessEqThan(n) {
    signal input in[2];
    signal output out;

    component lt = LessThan(n);
    lt.in[0] <== in[0];
    lt.in[1] <== in[1] + 1;
    out <== lt.out;
}

template GreaterThan(n) {
    signal input in[2];
    signal output out;

    component lt = LessThan(n);
    lt.in[0] <== in[1];
    lt.in[1] <== in[0];
    out <== lt.out;
}

template GreaterEqThan(n) {
    signal input in[2];
    signal output out;

    component lt = LessThan(n);
    lt.in[0] <== in[0];
    lt.in[1] <== in[1];
    out <== 1 - lt.out;
}

template Mux2() {
    signal input sel;
    signal input in[2];
    signal output out;

    out <== in[0] + sel * (in[1] - in[0]);
}

template IntegerDivision(bits) {
    signal input dividend;
    signal input divisor;
    signal output quotient;
    signal output remainder;

    quotient <-- dividend \ divisor;
    remainder <-- dividend % divisor;

    dividend === quotient * divisor + remainder;

    component remainderCheck = LessThan(bits);
    remainderCheck.in[0] <== remainder;
    remainderCheck.in[1] <== divisor;
    remainderCheck.out === 1;
}

template DecisionTreeClassifier() {
    signal input features[8];
    signal output intensity;

    var BITS = 16;

    var WRIST_ACCEL_STD = 6;
    var CHEST_ACCEL_STD = 1;
    var WRIST_GYRO_MEAN = 7;
    var ANKLE_ACCEL_STD = 3;
    var WRIST_ACCEL_MEAN = 5;
    var ANKLE_ACCEL_MEAN = 2;
    var ANKLE_GYRO_MEAN = 4;

    var REST = 0;
    var MODERATE = 1;
    var VIGOROUS = 2;

    component cmp0 = LessEqThan(BITS);
    cmp0.in[0] <== features[WRIST_ACCEL_STD];
    cmp0.in[1] <== 33;

    component cmp1 = LessEqThan(BITS);
    cmp1.in[0] <== features[WRIST_ACCEL_STD];
    cmp1.in[1] <== 26;

    component cmp4 = LessEqThan(BITS);
    cmp4.in[0] <== features[CHEST_ACCEL_STD];
    cmp4.in[1] <== 520;

    component cmp5 = LessEqThan(BITS);
    cmp5.in[0] <== features[WRIST_GYRO_MEAN];
    cmp5.in[1] <== 104;

    component cmp6 = LessEqThan(BITS);
    cmp6.in[0] <== features[ANKLE_ACCEL_STD];
    cmp6.in[1] <== 87;

    component cmp7 = LessEqThan(BITS);
    cmp7.in[0] <== features[WRIST_ACCEL_MEAN];
    cmp7.in[1] <== 1005;

    component cmp10 = LessEqThan(BITS);
    cmp10.in[0] <== features[ANKLE_ACCEL_MEAN];
    cmp10.in[1] <== 1268;

    component cmp13 = LessEqThan(BITS);
    cmp13.in[0] <== features[ANKLE_GYRO_MEAN];
    cmp13.in[1] <== 120;

    component cmp14 = LessEqThan(BITS);
    cmp14.in[0] <== features[ANKLE_GYRO_MEAN];
    cmp14.in[1] <== 82;

    component cmp17 = LessEqThan(BITS);
    cmp17.in[0] <== features[ANKLE_GYRO_MEAN];
    cmp17.in[1] <== 126;

    signal node7_result;
    component mux7 = Mux2();
    mux7.sel <== cmp7.out;
    mux7.in[0] <== MODERATE;
    mux7.in[1] <== VIGOROUS;
    node7_result <== mux7.out;

    signal node10_result;
    component mux10 = Mux2();
    mux10.sel <== cmp10.out;
    mux10.in[0] <== MODERATE;
    mux10.in[1] <== VIGOROUS;
    node10_result <== mux10.out;

    signal node6_result;
    component mux6 = Mux2();
    mux6.sel <== cmp6.out;
    mux6.in[0] <== node10_result;
    mux6.in[1] <== node7_result;
    node6_result <== mux6.out;

    signal node14_result;
    component mux14 = Mux2();
    mux14.sel <== cmp14.out;
    mux14.in[0] <== MODERATE;
    mux14.in[1] <== VIGOROUS;
    node14_result <== mux14.out;

    signal node17_result;
    node17_result <== VIGOROUS;

    signal node13_result;
    component mux13 = Mux2();
    mux13.sel <== cmp13.out;
    mux13.in[0] <== node17_result;
    mux13.in[1] <== node14_result;
    node13_result <== mux13.out;

    signal node5_result;
    component mux5 = Mux2();
    mux5.sel <== cmp5.out;
    mux5.in[0] <== node13_result;
    mux5.in[1] <== node6_result;
    node5_result <== mux5.out;

    signal node20_result;
    node20_result <== VIGOROUS;

    signal node4_result;
    component mux4 = Mux2();
    mux4.sel <== cmp4.out;
    mux4.in[0] <== node20_result;
    mux4.in[1] <== node5_result;
    node4_result <== mux4.out;

    signal node1_result;
    node1_result <== REST;

    signal node0_result;
    component mux0 = Mux2();
    mux0.sel <== cmp0.out;
    mux0.in[0] <== node4_result;
    mux0.in[1] <== node1_result;
    node0_result <== mux0.out;

    intensity <== node0_result;
}

template TimestampConsistencyChecker(numReadings, bitsTimestamp, maxGapSeconds) {
    signal input timestamps[numReadings];
    signal output isConsistent;

    component isIncreasing[numReadings - 1];
    component gapWithinLimit[numReadings - 1];

    signal increasingResults[numReadings - 1];
    signal gapResults[numReadings - 1];
    signal gaps[numReadings - 1];
    signal pairValid[numReadings - 1];
    signal runningValid[numReadings];

    runningValid[0] <== 1;

    for (var i = 0; i < numReadings - 1; i++) {
        isIncreasing[i] = GreaterThan(bitsTimestamp);
        isIncreasing[i].in[0] <== timestamps[i + 1];
        isIncreasing[i].in[1] <== timestamps[i];
        increasingResults[i] <== isIncreasing[i].out;

        gaps[i] <== timestamps[i + 1] - timestamps[i];

        gapWithinLimit[i] = LessEqThan(bitsTimestamp);
        gapWithinLimit[i].in[0] <== gaps[i];
        gapWithinLimit[i].in[1] <== maxGapSeconds;
        gapResults[i] <== gapWithinLimit[i].out;

        pairValid[i] <== increasingResults[i] * gapResults[i];
        runningValid[i + 1] <== runningValid[i] * pairValid[i];
    }

    isConsistent <== runningValid[numReadings - 1];
}

template SessionClassifier(numReadings, bitsTimestamp, maxGapSeconds) {
    signal input timestamps[numReadings];
    signal input features[8];

    signal output isConsistent;
    signal output intensity;
    signal output durationMinutes;

    component timestampChecker = TimestampConsistencyChecker(numReadings, bitsTimestamp, maxGapSeconds);
    for (var i = 0; i < numReadings; i++) {
        timestampChecker.timestamps[i] <== timestamps[i];
    }
    isConsistent <== timestampChecker.isConsistent;

    component classifier = DecisionTreeClassifier();
    for (var i = 0; i < 8; i++) {
        classifier.features[i] <== features[i];
    }
    intensity <== classifier.intensity;

    signal totalSeconds;
    totalSeconds <== timestamps[numReadings - 1] - timestamps[0];

    component divider = IntegerDivision(bitsTimestamp);
    divider.dividend <== totalSeconds;
    divider.divisor <== 60;
    durationMinutes <== divider.quotient;
}

template ZkMLFitnessClassifier(numSessions, numReadingsPerSession) {
    var WHO_GOAL_MINUTES = 150;
    var MAX_GAP_SECONDS = 60;
    var BITS_TIMESTAMP = 32;

    var REST = 0;
    var MODERATE = 1;
    var VIGOROUS = 2;

    signal input weekTimestamp;
    signal input timestamps[numSessions][numReadingsPerSession];
    signal input features[numSessions][8];

    signal output goalAchieved;
    signal output verifiedWeekTimestamp;
    signal output totalEquivalentMinutes;
    signal output allSessionsConsistent;

    component sessions[numSessions];

    signal sessionConsistent[numSessions];
    signal sessionIntensity[numSessions];
    signal sessionDuration[numSessions];

    signal validDuration[numSessions];
    signal isModerate[numSessions];
    signal isVigorous[numSessions];
    signal moderateMinutes[numSessions];
    signal vigorousMinutes[numSessions];

    component cmpMod[numSessions];
    component cmpRest[numSessions];
    component cmpVig[numSessions];

    signal runningModerate[numSessions + 1];
    signal runningVigorous[numSessions + 1];
    signal runningConsistent[numSessions + 1];

    runningModerate[0] <== 0;
    runningVigorous[0] <== 0;
    runningConsistent[0] <== 1;

    for (var i = 0; i < numSessions; i++) {
        sessions[i] = SessionClassifier(numReadingsPerSession, BITS_TIMESTAMP, MAX_GAP_SECONDS);

        for (var j = 0; j < numReadingsPerSession; j++) {
            sessions[i].timestamps[j] <== timestamps[i][j];
        }
        for (var j = 0; j < 8; j++) {
            sessions[i].features[j] <== features[i][j];
        }

        sessionConsistent[i] <== sessions[i].isConsistent;
        sessionIntensity[i] <== sessions[i].intensity;
        sessionDuration[i] <== sessions[i].durationMinutes;

        validDuration[i] <== sessionConsistent[i] * sessionDuration[i];

        cmpMod[i] = LessThan(4);
        cmpMod[i].in[0] <== sessionIntensity[i];
        cmpMod[i].in[1] <== VIGOROUS;

        cmpRest[i] = GreaterThan(4);
        cmpRest[i].in[0] <== sessionIntensity[i];
        cmpRest[i].in[1] <== REST;

        isModerate[i] <== cmpMod[i].out * cmpRest[i].out;

        cmpVig[i] = GreaterEqThan(4);
        cmpVig[i].in[0] <== sessionIntensity[i];
        cmpVig[i].in[1] <== VIGOROUS;
        isVigorous[i] <== cmpVig[i].out;

        moderateMinutes[i] <== isModerate[i] * validDuration[i];
        vigorousMinutes[i] <== isVigorous[i] * validDuration[i];

        runningModerate[i + 1] <== runningModerate[i] + moderateMinutes[i];
        runningVigorous[i + 1] <== runningVigorous[i] + vigorousMinutes[i];
        runningConsistent[i + 1] <== runningConsistent[i] * sessionConsistent[i];
    }

    signal totalModerate;
    signal totalVigorous;
    totalModerate <== runningModerate[numSessions];
    totalVigorous <== runningVigorous[numSessions];

    totalEquivalentMinutes <== totalModerate + totalVigorous * 2;

    allSessionsConsistent <== runningConsistent[numSessions];

    component goalCheck = GreaterEqThan(16);
    goalCheck.in[0] <== totalEquivalentMinutes;
    goalCheck.in[1] <== WHO_GOAL_MINUTES;

    signal goalMet;
    goalMet <== goalCheck.out;

    goalAchieved <== goalMet * allSessionsConsistent;

    verifiedWeekTimestamp <== weekTimestamp;
}

component main {public [weekTimestamp]} = ZkMLFitnessClassifier(10, 12);
