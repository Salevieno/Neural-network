
plugins {
    // Apply the application plugin to add support for building a CLI application in Java.
    application
}

repositories {
    // Use Maven Central for resolving dependencies.
    mavenCentral()
}

dependencies {
    // Use JUnit Jupiter for testing.
    testImplementation(libs.junit.jupiter)

    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    implementation("org.ejml:ejml-all:0.42")
    implementation("org.ejml:ejml-core:0.42")
    implementation("org.ejml:ejml-simple:0.42")

    implementation("com.google.code.gson:gson:2.12.1")
    

    // This dependency is used by the application.
    implementation(libs.guava)

    implementation(files("libs/libGraphics.jar"))
    implementation(files("libs/libChart.jar"))
    implementation(files("libs/libUtil.jar"))
}

// Apply a specific Java toolchain to ease working on different environments.
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

application {
    // Define the main class for the application.
    mainClass = "neural.network.App"
}

tasks.named<Test>("test") {
    // Use JUnit Platform for unit tests.
    useJUnitPlatform()
}
